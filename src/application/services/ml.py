from src.infrastructure.ml_models import attractiveness_model  # , looak_a_like_finder
from src.interfaces.api.schemas import Inference


class MLService:
    @staticmethod
    def get_attractiveness(image: bytes) -> float | None:
        if not image:
            return None
        try:
            predicted_attractiveness = attractiveness_model.predict(image)
        except:
            return None
        return round(predicted_attractiveness, 4)

    @staticmethod
    def get_celebrities(image: bytes) -> list[int] | None:
        return []
        if not image:
            return None
        try:
            celebrities = lookup_a_like_finder.predict(image)
        except:
            return None
        return celebrities

    @staticmethod
    def create_inference(user_id: int, image: bytes) -> Inference:
        celebritty_ids = MLService.get_celebrities(image)
        attractiveness = MLService.get_attractiveness(image)
