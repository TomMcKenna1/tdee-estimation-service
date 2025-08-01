# tdee_estimation_service/config.py

MIN_PLAUSIBLE_INTAKE_RATIO = 0.6
# --- Kalman Filter Tuning Parameters ---

# Estimated variance of the daily process noise.
# How much we expect the user's true TDEE to fluctuate daily, squared.
# A value of 50^2 means we think TDEE naturally varies by about 50 kcal.
PROCESS_NOISE_VAR_TDEE = 50.0**2
# How much we expect the user's calory tracking to fluctuate daily, squared.
# A value of 200^2 means we think caloric intake varies by around 200 kcal.
CALORIE_INTAKE_UNCERTAINTY_VAR = 200.0**2

# Estimated variance of the user's weight measurement, squared.
# A value of 0.5^2 means we trust the user's scale to be accurate within ~0.5 kg.
MEASUREMENT_NOISE_VAR_WEIGHT = 0.5**2

# --- Initial State Configuration ---

# Initial uncertainty (variance) for the activity multiplier. High because we're guessing.
INITIAL_ACTIVITY_MULTIPLIER_UNCERTAINTY = 0.5**2

# Initial uncertainty (variance) for weight. Low because it comes from a measurement.
INITIAL_WEIGHT_UNCERTAINTY = 0.1**2

# Confidence interval for bounds (e.g., 1.96 for 95% confidence).
Z_SCORE_FOR_BOUNDS = 1.96

# --- Constants ---
KCAL_PER_KG_BODY_WEIGHT = 7700.0

# --- Initial Guess Mapping ---
# Maps user-reported activity level to a starting activity multiplier.
ACTIVITY_LEVEL_MAPPING = {
    "sedentary": 1.2,
    "lightly_active": 1.375,
    "moderately_active": 1.55,
    "very_active": 1.725,
    "extra_active": 1.9,
}
DEFAULT_ACTIVITY_LEVEL = 1.2
