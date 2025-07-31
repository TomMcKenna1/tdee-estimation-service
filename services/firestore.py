import logging
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
from models.user import UserInDB
from models.tdee_history import TDEEHistory

MOCK_DB = {
    "users": {
        "user_123": {
            "uid": "user_123",
            "name": "Jane Doe",
            "email": "jane@example.com",
            "createdAt": datetime(2025, 7, 1, 0, 0),
            "onboardingComplete": True,
            "profile": {
                "sex": "male",
                "age": 25,
                "heightCm": 188,
                "weightKg": 76,
                "goal": "maintain_weight",
                "activityLevel": "moderately_active",
            },
        }
    },
    "meals": {
        "meal_1": {
            "uid": "user_123",
            "createdAt": datetime(2025, 7, 28, 8, 0),
            "status": "complete",
            "data": {"nutrientProfile": {"energy": 500}},
        },
        "meal_2": {
            "uid": "user_123",
            "createdAt": datetime(2025, 7, 28, 13, 0),
            "status": "complete",
            "data": {"nutrientProfile": {"energy": 800}},
        },
        "meal_3": {
            "uid": "user_123",
            "createdAt": datetime(2025, 7, 28, 19, 0),
            "status": "complete",
            "data": {"nutrientProfile": {"energy": 850}},
        },
        "meal_4": {
            "uid": "user_123",
            "createdAt": datetime(2025, 7, 29, 9, 0),
            "status": "pending",
            "data": {"nutrientProfile": {"energy": 400}},
        },
    },
    "weight_logs": {
        "log_1": {
            "uid": "user_123",
            "createdAt": datetime(2025, 7, 26, 7, 0),
            "weight_kg": 75.5,
        }
    },
    "tdeeHistory": {},
}


class FirestoreService:
    """Handles all communication with the Firestore database."""

    def __init__(self):
        logging.info("FirestoreService initialized (using mock data).")

    def get_all_users(self) -> List[UserInDB]:
        """Fetches all users from the 'users' collection."""
        return [UserInDB(**u) for u in MOCK_DB["users"].values()]

    def get_caloric_intake_for_date_range(
        self, uid: str, start_date: date, end_date: date
    ) -> Dict[date, float]:
        """Fetches and sums daily caloric intake for a user."""
        results = {}
        for d in range((end_date - start_date).days + 1):
            current_date = start_date + timedelta(days=d)
            results[current_date] = 0.0

        for meal in MOCK_DB["meals"].values():
            if meal["uid"] == uid and meal["status"] == "complete":
                meal_date = meal["createdAt"].date()
                if start_date <= meal_date <= end_date:
                    results[meal_date] += meal["data"]["nutrientProfile"]["energy"]
        return results

    def get_weight_logs_for_date_range(
        self, uid: str, start_date: date, end_date: date
    ) -> Dict[date, float]:
        """Fetches daily weight measurements for a user."""
        results = {}
        for log in MOCK_DB["weight_logs"].values():
            if log["uid"] == uid:
                log_date = log["createdAt"].date()
                if start_date <= log_date <= end_date:
                    results[log_date] = log["weight_kg"]
        return results

    def get_last_tdee_history(self, uid: str) -> Optional[TDEEHistory]:
        """Gets the most recent TDEE history entry for a user."""
        user_history = [h for h in MOCK_DB["tdeeHistory"].values() if h["uid"] == uid]
        if not user_history:
            return None

        latest_entry = max(user_history, key=lambda x: x["date"])
        return TDEEHistory.model_validate(latest_entry)

    def save_tdee_history(self, history_entry: TDEEHistory):
        """Saves a TDEE history entry to Firestore."""
        doc_id = f"{history_entry.uid}_{history_entry.date.isoformat()}"
        data = history_entry.model_dump(by_alias=True)
        logging.info(
            f"Saving TDEE history for {doc_id}: TDEE {data['estimatedTdeeKcal']:.0f} kcal"
        )
        MOCK_DB["tdeeHistory"][doc_id] = data

    def get_DB(self):
        return MOCK_DB
