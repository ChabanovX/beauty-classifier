from fastapi import APIRouter, UploadFile, HTTPException
from .schemas import AttractivenessPrediction
from ..services.attractiveness import AttractivenessService

attractiveness_router = APIRouter()


@attractiveness_router.post("/predict")
async def predict(image_file: UploadFile):
    if not image_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
        )

    image_bytes = await image_file.read()

    score = AttractivenessService.get_attractiveness(image_bytes)

    if not score:
        raise HTTPException(status_code=400, detail="Could not process image")

    return AttractivenessPrediction(prediction=score)
