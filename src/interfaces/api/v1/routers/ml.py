from fastapi import APIRouter, UploadFile, HTTPException, status
from ..schemas import AttractivenessPrediction
from src.application.services.ml import MLService

ml_router = APIRouter(prefix="/ml", tags=["ML"])


@ml_router.post("/attractiveness")
async def predict(image_file: UploadFile) -> AttractivenessPrediction:
    if not image_file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file")
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    score = MLService.get_attractiveness(image_bytes)

    if not score:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process image",
        )

    return AttractivenessPrediction(prediction=score)
