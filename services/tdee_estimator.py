import numpy as np
import config
from utils.bmr_calculator import calculate_mifflin_st_jeor_bmr
from models.profile import Sex


class TDEEEstimator:
    """
    Implements a Kalman Filter to estimate TDEE from daily caloric intake
    and periodic weight measurements.
    """

    def __init__(
        self, user_profile: dict, prev_state: np.ndarray, prev_covariance: np.ndarray
    ):
        self.sex = Sex(user_profile["sex"])
        self.age = user_profile["age"]
        self.height_cm = user_profile["height_cm"]
        self.state = prev_state
        self.covariance = prev_covariance
        self.F = np.array([[1.0, -1.0 / config.KCAL_PER_KG_BODY_WEIGHT], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.R = np.array([[config.MEASUREMENT_NOISE_VAR_WEIGHT]])

    def predict(self, kcal_intake: float):
        prev_weight, prev_tdee = self.state
        weight_change = (kcal_intake - prev_tdee) / config.KCAL_PER_KG_BODY_WEIGHT
        predicted_weight = prev_weight + weight_change
        predicted_tdee = prev_tdee
        self.state = np.array([predicted_weight, predicted_tdee])
        weight_variance_from_calories = config.CALORIE_INTAKE_UNCERTAINTY_VAR / (
            config.KCAL_PER_KG_BODY_WEIGHT**2
        )
        Q = np.array(
            [
                [weight_variance_from_calories, 0],
                [0, config.PROCESS_NOISE_VAR_TDEE],
            ]
        )
        self.covariance = self.F @ self.covariance @ self.F.T + Q

    def update(self, measured_weight_kg: float):
        y = measured_weight_kg - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.covariance = (np.identity(2) - K @ self.H) @ self.covariance

    def get_bounds(self) -> tuple[float, float]:
        tdee_variance = self.covariance[1, 1]
        tdee_std_dev = np.sqrt(tdee_variance)
        margin_of_error = config.Z_SCORE_FOR_BOUNDS * tdee_std_dev
        current_tdee = self.state[1]
        return current_tdee - margin_of_error, current_tdee + margin_of_error

    def get_weight_bounds(self) -> tuple[float, float]:
        """
        Calculates the upper and lower confidence bounds for the weight estimate.
        """
        weight_variance = self.covariance[0, 0]
        weight_std_dev = np.sqrt(weight_variance)
        margin_of_error = config.Z_SCORE_FOR_BOUNDS * weight_std_dev
        current_weight = self.state[0]
        lower_bound = current_weight - margin_of_error
        upper_bound = current_weight + margin_of_error
        return lower_bound, upper_bound