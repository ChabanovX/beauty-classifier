from pydantic import BaseModel, ConfigDict

class Base(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

class AttractivenessPrediction(Base):
    prediction: float