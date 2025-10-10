from .config import config
from .v1.routes import attractiveness_router
from .models import attractiveness_model
# from .v1.users import router as UserRouter
from .v1.auth import router as AuthRouter
# from fastapi.middleware.cors import CORSMiddleware
from .db.db_connector import connect, close

from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# сначала безопасно попытаться импортировать роутеры
UserRouter = None
AuthRouter = None
# try:
#     from .v1.users import router as UserRouter
#     logger.info("Imported UserRouter")
# except Exception as e:
#     logger.exception("Failed to import UserRouter: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await connect()
        attractiveness_model.load()
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    yield
    await close()


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
# app.include_router(UserRouter, prefix="/users")
app.include_router(AuthRouter, prefix="/auth")
# app.include_router(UserRouter, tags=["User"])


# @app.on_event("startup")
# async def _print_routes():
#     logger.info("=== Registered routes ===")
#     for route in app.routes:
#         try:
#             logger.info("%s %s %s", route.path, getattr(route, "methods", None), getattr(route, "name", None))
#         except Exception:
#             logger.exception("Error printing route")
#     logger.info("=========================")


if __name__ == "__main__":
    uvicorn.run(app, host=config.api_host, port=config.api_port)
