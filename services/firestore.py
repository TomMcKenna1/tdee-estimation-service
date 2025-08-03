# services/firestore.py
import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Optional, AsyncGenerator

import firebase_admin
from firebase_admin import credentials, firestore_async
from google.cloud.firestore_v1.async_client import AsyncClient
from google.cloud.firestore_v1.base_query import FieldFilter

from models.user import UserInDB
from models.tdee_history import TDEEHistory
from models.weight_forecast import WeightForecast  # New import


def initialize_firebase_app():
    if not firebase_admin._apps:
        cred_path = os.path.join(
            os.path.dirname(__file__), "..", "service-account.json"
        )
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"Service account key not found at {cred_path}.")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        logging.info("Firebase Admin SDK initialized successfully.")


class FirestoreService:
    def __init__(self):
        initialize_firebase_app()
        self.db: AsyncClient = firestore_async.client()

    async def get_all_users(
        self, page_size: int = 1000
    ) -> AsyncGenerator[UserInDB, None]:
        users_ref = self.db.collection("users")
        cursor = None
        while True:
            query = users_ref.order_by("__name__").limit(page_size)
            if cursor:
                query = query.start_after(cursor)
            docs = await query.get()
            if not docs:
                break
            for doc in docs:
                yield UserInDB(uid=doc.id, **doc.to_dict())
            cursor = docs[-1]

    async def get_caloric_intake_for_date_range(
        self, uid: str, start_date: date, end_date: date
    ) -> Dict[date, float]:
        results = {
            start_date + timedelta(days=d): 0.0
            for d in range((end_date - start_date).days + 1)
        }
        start_utc = datetime.combine(
            start_date, datetime.min.time(), tzinfo=timezone.utc
        )
        end_utc = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        meals_ref = self.db.collection("users").document(uid).collection("mealLogs")
        query = (
            meals_ref.where(filter=FieldFilter("createdAt", ">=", start_utc))
            .where(filter=FieldFilter("createdAt", "<=", end_utc))
            .where(filter=FieldFilter("status", "==", "complete"))
        )
        docs = await query.get()
        for doc in docs:
            meal_data = doc.to_dict()
            meal_date = meal_data["createdAt"].date()
            if meal_date in results:
                results[meal_date] += (
                    meal_data.get("data", {})
                    .get("nutrientProfile", {})
                    .get("energy", 0)
                )
        return results

    async def get_weight_logs_for_date_range(
        self, uid: str, start_date: date, end_date: date
    ) -> Dict[date, float]:
        results = {}
        start_utc = datetime.combine(
            start_date, datetime.min.time(), tzinfo=timezone.utc
        )
        end_utc = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        logs_ref = self.db.collection("users").document(uid).collection("weightLogs")
        query = logs_ref.where(filter=FieldFilter("date", ">=", start_utc)).where(
            filter=FieldFilter("date", "<=", end_utc)
        )
        docs = await query.get()
        daily_logs: Dict[date, List[float]] = {}
        for doc in docs:
            log_data = doc.to_dict()
            log_date = log_data["date"].date()
            daily_logs.setdefault(log_date, []).append(log_data.get("weightKg", 0.0))
        for day, weights in daily_logs.items():
            if weights:
                results[day] = sum(weights) / len(weights)
        return results

    async def get_last_tdee_history(self, uid: str) -> Optional[TDEEHistory]:
        history_ref = (
            self.db.collection("users").document(uid).collection("tdeeHistory")
        )
        query = history_ref.order_by("date", direction="DESCENDING").limit(1)
        docs = await query.get()
        if not docs:
            return None
        return TDEEHistory.model_validate(docs[0].to_dict())

    async def save_tdee_history_batch(self, history_entries: List[TDEEHistory]):
        if not history_entries:
            return
        batch = self.db.batch()
        user_id = history_entries[0].uid
        history_ref = (
            self.db.collection("users").document(user_id).collection("tdeeHistory")
        )
        for entry in history_entries:
            doc_id = entry.date.isoformat()
            doc_ref = history_ref.document(doc_id)
            batch.set(doc_ref, entry.model_dump(by_alias=True))
        await batch.commit()

    async def save_weight_forecast(
        self, uid: str, forecast_entries: List[WeightForecast]
    ):
        """
        Saves a batch of weight forecasts by overwriting existing documents.
        This is more efficient than deleting and recreating the subcollection.
        """
        if not forecast_entries:
            return

        forecast_ref = (
            self.db.collection("users").document(uid).collection("weightForecast")
        )
        batch = self.db.batch()

        for entry in forecast_entries:
            doc_id = entry.date.date().isoformat()
            doc_ref = forecast_ref.document(doc_id)
            batch.set(doc_ref, entry.model_dump(by_alias=True))

        await batch.commit()
        logging.info(
            f"Saved/Overwrote {len(forecast_entries)} weight forecasts for user {uid}."
        )
