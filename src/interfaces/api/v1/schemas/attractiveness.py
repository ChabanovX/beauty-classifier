from pydantic import Field

from .base import Base


class AttractivenessPrediction(Base):
    prediction: float = Field(examples=[4.0], description="Attractiveness score")
