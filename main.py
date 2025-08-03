# main.py
import logging
import asyncio
from typing import Dict, List
import numpy as np
from datetime import date, datetime, timedelta, timezone
import config
from services.firestore import FirestoreService
from services.tdee_estimator import TDEEEstimator
from utils.bmr_calculator import calculate_mifflin_st_jeor_bmr
from models.user import UserInDB
from models.tdee_history import TDEEHistory
from models.weight_forecast import WeightForecast

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_initial_state(user: UserInDB) -> tuple[np.ndarray, np.ndarray]:
    profile = user.profile
    initial_weight = user.current_weight_kg
    activity_level_str = (
        profile.activity_level.value if profile.activity_level else "sedentary"
    )
    initial_multiplier = config.ACTIVITY_LEVEL_MAPPING.get(
        activity_level_str, config.DEFAULT_ACTIVITY_LEVEL
    )
    bmr = calculate_mifflin_st_jeor_bmr(
        profile.sex, profile.age, profile.height_cm, initial_weight
    )
    initial_tdee = bmr * initial_multiplier
    X_0 = np.array([initial_weight, initial_tdee])
    P_0 = np.array(
        [
            [config.INITIAL_WEIGHT_UNCERTAINTY, 0],
            [0, (bmr * config.INITIAL_ACTIVITY_MULTIPLIER_UNCERTAINTY) ** 2],
        ]
    )
    return X_0, P_0


def _get_fallback_intake(
    caloric_data: Dict[date, float], up_to_date: date, last_known_tdee: float
) -> float:
    relevant_intakes = [
        intake
        for d, intake in caloric_data.items()
        if intake > 0 and 0 < (up_to_date - d).days <= 7
    ]
    if relevant_intakes:
        return sum(relevant_intakes) / len(relevant_intakes)
    return last_known_tdee


async def process_user(fs: FirestoreService, user: UserInDB):
    logging.info(f"--- Processing user: {user.uid} ({user.email}) ---")

    if not all(
        [
            user.profile,
            user.profile.sex,
            user.profile.age,
            user.profile.height_cm,
            user.current_weight_kg,
        ]
    ):
        logging.warning(f"User {user.uid} is missing essential profile data. Skipping.")
        return

    last_history = await fs.get_last_tdee_history(user.uid)
    processing_end_date = date.today() - timedelta(days=2)

    if last_history:
        start_date = last_history.date.date() + timedelta(days=1)
        state = np.array(
            [last_history.estimated_weight_kg, last_history.estimated_tdee_kcal]
        )
        covariance = np.array(last_history.covariance_matrix).reshape(2, 2)
    else:
        start_date = user.created_at.date()
        state, covariance = get_initial_state(user)

    if start_date > processing_end_date:
        logging.info(
            f"User {user.uid} is already up-to-date. Generating forecast only."
        )
    else:
        logging.info(
            f"Processing date range: {start_date.isoformat()} to {processing_end_date.isoformat()}"
        )
        caloric_data = await fs.get_caloric_intake_for_date_range(
            user.uid, start_date - timedelta(days=91), processing_end_date
        )
        weight_data = await fs.get_weight_logs_for_date_range(
            user.uid, start_date, processing_end_date
        )
        current_date = start_date
        daily_updates: List[TDEEHistory] = []
        while current_date <= processing_end_date:
            prev_day = current_date - timedelta(days=1)
            prev_day_intake = caloric_data.get(prev_day)
            last_known_tdee = state[1]
            is_implausible = prev_day_intake is not None and 0 < prev_day_intake < (
                last_known_tdee * config.MIN_PLAUSIBLE_INTAKE_RATIO
            )
            if prev_day_intake is None or prev_day_intake == 0.0 or is_implausible:
                prev_day_intake = _get_fallback_intake(
                    caloric_data, prev_day, last_known_tdee
                )

            estimator = TDEEEstimator(user.profile.model_dump(), state, covariance)
            estimator.predict(kcal_intake=prev_day_intake)

            is_prediction = True
            if current_date in weight_data:
                is_prediction = False
                estimator.update(weight_data[current_date])

            state = estimator.state
            covariance = estimator.covariance
            lower_bound, upper_bound = estimator.get_bounds()
            current_datetime = datetime.combine(
                current_date, datetime.min.time(), tzinfo=timezone.utc
            )
            history_entry = TDEEHistory(
                uid=user.uid,
                date=current_datetime,
                estimated_tdee_kcal=state[1],
                lower_bound_kcal=lower_bound,
                upper_bound_kcal=upper_bound,
                is_prediction=is_prediction,
                estimated_weight_kg=state[0],
                activity_multiplier=state[1]
                / calculate_mifflin_st_jeor_bmr(
                    user.profile.sex, user.profile.age, user.profile.height_cm, state[0]
                ),
                covariance_matrix=covariance.flatten().tolist(),
            )
            daily_updates.append(history_entry)
            current_date += timedelta(days=1)
        if daily_updates:
            await fs.save_tdee_history_batch(daily_updates)

    ## --- Generate 14-Day Weight Forecast ---
    logging.info(f"Generating 14-day weight forecast for user {user.uid}.")
    caloric_data = await fs.get_caloric_intake_for_date_range(
        user.uid, processing_end_date - timedelta(days=90), processing_end_date
    )

    # Can add this later
    assumed_intake_goal = None

    final_daily_averages = None
    if assumed_intake_goal:
        logging.info(
            f"User has a calorie goal set. Using {assumed_intake_goal:.0f} kcal for a linear forecast."
        )
    else:
        logging.info(
            "No calorie goal set. Using 3-month historical day-of-week averages for a non-linear forecast."
        )
        daily_intake_averages = {i: [] for i in range(7)}
        for day, intake in caloric_data.items():
            if intake > 0 and (processing_end_date - day).days <= 90:
                daily_intake_averages[day.weekday()].append(intake)
        fallback_intake = state[1]
        final_daily_averages = [
            sum(intakes) / len(intakes) if intakes else fallback_intake
            for intakes in daily_intake_averages.values()
        ]

    forecast_estimator = TDEEEstimator(user.profile.model_dump(), state, covariance)
    future_forecasts: List[WeightForecast] = []
    forecast_start_date = processing_end_date + timedelta(days=1)

    for i in range(14):
        prediction_date = forecast_start_date + timedelta(days=i)

        if assumed_intake_goal:
            intake_for_day = assumed_intake_goal
        else:
            day_of_week = prediction_date.weekday()
            intake_for_day = final_daily_averages[day_of_week]

        forecast_estimator.predict(kcal_intake=intake_for_day)

        predicted_weight = forecast_estimator.state[0]
        lower_bound, upper_bound = forecast_estimator.get_weight_bounds()

        prediction_datetime = datetime.combine(
            prediction_date, datetime.min.time(), tzinfo=timezone.utc
        )

        future_forecasts.append(
            WeightForecast(
                date=prediction_datetime,
                predictedWeightKg=predicted_weight,
                lowerBoundKg=lower_bound,
                upperBoundKg=upper_bound,
            )
        )

    await fs.save_weight_forecast(user.uid, future_forecasts)


async def run_daily_job():
    logging.info("Starting TDEE estimation daily job.")
    firestore_service = FirestoreService()
    user_count = 0
    async for user in firestore_service.get_all_users():
        user_count += 1
        try:
            await process_user(firestore_service, user)
        except Exception as e:
            logging.error(
                f"An unexpected error occurred while processing user {user.uid}: {e}",
                exc_info=True,
            )
    logging.info(f"Processed a total of {user_count} user(s).")
    logging.info("TDEE estimation daily job finished.")


if __name__ == "__main__":
    asyncio.run(run_daily_job())
