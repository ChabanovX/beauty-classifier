from fastapi import APIRouter, UploadFile, HTTPException
from src.infrastructure.schemas.attractiveness import AttractivenessPrediction
from src.application.services.ml import MLService

ml_router = APIRouter(tags=["ML"])


@ml_router.post("/attractiveness/predict")
async def predict(image_file: UploadFile):
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    score = MLService.get_attractiveness(image_bytes)

    if not score:
        raise HTTPException(status_code=400, detail="Could not process image")

    return AttractivenessPrediction(prediction=score)
