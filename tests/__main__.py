from fastapi.testclient import TestClient

from api import app
from api.models import attractiveness_model

client = TestClient(app)


def load_model():
    global attractiveness_model
    attractiveness_model.load()
    assert attractiveness_model._model is not None, "Model could not be loaded"
    print("Model loaded")


def evaluate_model():
    global attractiveness_model
    if not attractiveness_model.loaded:
        load_model()
    attractiveness_model.evaluate()
    print("Model evaluated")


def test_predict_endpoint():
    global attractiveness_model
    if not attractiveness_model.loaded:
        load_model()
    image_file = ("test.jpg", b"test", "image/jpeg")
    response = client.post("/predict", files={"image_file": image_file})
    response.raise_for_status()
    assert response.content, "Response is empty"
    assert response.json().get("prediction", None) is not None, "Prediction is None"
    print("Prediction endpoint tested")


if __name__ == "__main__":
    load_model()
    test_predict_endpoint()
    evaluate_model()
    print("Tests passed")
