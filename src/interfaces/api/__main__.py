from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import config
from src.infrastructure.database.core import db_engine_lifespan
from src.infrastructure.ml_models import load_models
from src.interfaces.api.v1 import routers


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    async with db_engine_lifespan():
        yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

for router in routers:
    app.include_router(router)
import logging

logging.info(__package__)

if __name__ == "__main__":
    uvicorn.run(
        f"{__package__}.__main__:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_config=config.logging.config,
    )
