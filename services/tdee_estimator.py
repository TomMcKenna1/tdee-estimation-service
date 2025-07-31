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
        """
        Initializes the estimator for a single time step.

        Args:
            user_profile: A dictionary containing user's 'sex', 'age', 'height_cm'.
            prev_state: The previous day's state vector [weight_kg, tdee_kcal].
            prev_covariance: The previous day's 2x2 covariance matrix.
        """
        self.sex = Sex(user_profile["sex"])
        self.age = user_profile["age"]
        self.height_cm = user_profile["height_cm"]

        self.state = prev_state
        self.covariance = prev_covariance

        self.F = np.array([[1.0, -1.0 / config.KCAL_PER_KG_BODY_WEIGHT], [0.0, 1.0]])
        # Observation matrix (we only observe weight)
        self.H = np.array([[1.0, 0.0]])
        # Measurement noise
        self.R = np.array([[config.MEASUREMENT_NOISE_VAR_WEIGHT]])

    def _get_bmr_and_jacobian(self) -> tuple[float, np.ndarray]:
        """Calculates BMR and its derivative w.r.t. state variables."""
        current_weight = self.state[0]
        bmr = calculate_mifflin_st_jeor_bmr(
            self.sex, self.age, self.height_cm, current_weight
        )

        # Jacobian of BMR w.r.t. state [weight, TDEE]
        # BMR depends only on weight (10 * w + ...). Derivative is 10.
        bmr_jacobian = np.array([10.0, 0.0])
        return bmr, bmr_jacobian

    def predict(self, kcal_intake: float):
        """
        Performs the prediction step of the Kalman Filter.
        Estimates today's state based on yesterday's state and intake.
        """
        prev_weight, prev_tdee = self.state

        # Predict next state
        weight_change = (kcal_intake - prev_tdee) / config.KCAL_PER_KG_BODY_WEIGHT
        predicted_weight = prev_weight + weight_change
        predicted_tdee = prev_tdee

        self.state = np.array([predicted_weight, predicted_tdee])

        # Predict next covariance
        # Process noise matrix Q
        weight_variance_from_calories = config.CALORIE_INTAKE_UNCERTAINTY_VAR / (config.KCAL_PER_KG_BODY_WEIGHT ** 2)
        Q = np.array(
            [
                [weight_variance_from_calories, 0],
                [0, config.PROCESS_NOISE_VAR_TDEE],
            ]
        )
        self.covariance = self.F @ self.covariance @ self.F.T + Q

    def update(self, measured_weight_kg: float):
        """
        Performs the update step of the Kalman Filter.
        Corrects the prediction with a new weight measurement.
        """
        # Kalman gain calc
        # Measurement residual
        y = measured_weight_kg - self.H @ self.state
        # Residual covariance
        S = self.H @ self.covariance @ self.H.T + self.R
        # Kalman Gain
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.identity(2) - K @ self.H) @ self.covariance

    def get_bounds(self) -> tuple[float, float]:
        """Calculates the upper and lower bounds for the TDEE estimate."""
        tdee_variance = self.covariance[1, 1]
        tdee_std_dev = np.sqrt(tdee_variance)
        margin_of_error = config.Z_SCORE_FOR_BOUNDS * tdee_std_dev

        current_tdee = self.state[1]
        return current_tdee - margin_of_error, current_tdee + margin_of_error
