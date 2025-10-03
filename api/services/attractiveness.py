from ..models import attractiveness_model


class AttractivenessService:
    @staticmethod
    def get_attractiveness(image: bytes | None) -> float | None:
        if not image:
            return None
        try:
            predicted_attractiveness = attractiveness_model.predict(image)
        except ValueError:
            return None
        return round(predicted_attractiveness, 4)
