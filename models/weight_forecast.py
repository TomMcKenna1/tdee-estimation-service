from datetime import datetime
from pydantic import BaseModel, Field


class WeightForecast(BaseModel):
    date: datetime = Field(..., description="The future date of the forecast.")
    predicted_weight_kg: float = Field(..., alias="predictedWeightKg")
    lower_bound_kg: float = Field(..., alias="lowerBoundKg")
    upper_bound_kg: float = Field(..., alias="upperBoundKg")

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}
