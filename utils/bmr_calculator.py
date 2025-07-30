from models.profile import Sex


def calculate_mifflin_st_jeor_bmr(
    sex: Sex, age: int, height_cm: float, weight_kg: float
) -> float:
    """Calculates BMR using the Mifflin-St Jeor equation."""
    if sex == Sex.MALE:
        return (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) + 5
    else:
        return (10.0 * weight_kg) + (6.25 * height_cm) - (5.0 * age) - 161
