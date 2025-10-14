import warnings
import logging.config

from pydantic import BaseModel
from pydantic_settings import BaseSettings, CliImplicitFlag, SettingsConfigDict

from .logconf import Logging


class DB(BaseModel):
    url: str
    dev_url: str
    connection_timeout: int
    pool_size: int
    pool_timeout: int
    connection_string: str | None = None


class API(BaseModel):
    host: str
    port: int
    reload: CliImplicitFlag[bool] = False


class ML(BaseModel):
    mlflow_tracking_url: str
    scut_data_path: str


class Auth(BaseModel):
    secret_key: str
    algorithm: str
    access_token_expire: int
    salt: str


class App(BaseModel):
    dev: CliImplicitFlag[bool] = True
    prod: CliImplicitFlag[bool] = False

    @property
    def env(self) -> str:
        return "DEV" if self.dev else "PROD"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env.example", ".env"],
        env_nested_delimiter="__",
        cli_implicit_flags=["app.dev", "app.prod", "api.reload"],
    )

    db: DB
    api: API
    ml: ML
    auth: Auth
    app: App
    logging: Logging

    def model_post_init(self, context):
        self.logging.level = "DEBUG" if self.app.dev else "INFO"
        self.logging.config = (
            self.logging.dev_config if self.app.dev else self.logging.prod_config
        )
        self.logging.file = (
            self.app.dev
            and self.logging.file.replace(".log", "_dev.log")
            or self.logging.file
        )
        self.db.connection_string = (
            self.app.dev
            and self.db.dev_url.replace("sqlite://", "sqlite+aiosqlite://")
            or self.db.url.replace("postgres://", "postgresql+asyncpg://")
        )


config = Config()

logging.config.dictConfig(config.logging.config)

# if config.app.dev:
#     logging.debug(
#         f"Loaded config: {config.model_dump_json(indent=2, exclude='logging')}"
#     )

warnings.simplefilter(action="ignore", category=FutureWarning)
