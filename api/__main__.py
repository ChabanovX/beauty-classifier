from .config import config
from .v1.routes import attractiveness_router
from .models.attractiveness.model import attractiveness_model

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        attractiveness_model.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(attractiveness_router, prefix="/attractiveness")

if __name__ == "__main__":
    uvicorn.run(app, host=config.api_host, port=config.api_port)
