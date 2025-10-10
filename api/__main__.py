from .config import config
from .v1.routes import attractiveness_router
from .models import attractiveness_model
from .v1.users import UserRouter
# from fastapi.middleware.cors import CORSMiddleware

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

# do we need to add CORSMiddleware?
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.include_router(attractiveness_router, prefix="/attractiveness")
# app.include_router(UserRouter, tags=["User"])

if __name__ == "__main__":
    uvicorn.run(app, host=config.api_host, port=config.api_port)
