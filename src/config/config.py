from typing import Literal
import warnings

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logconf import Logging


class DB(BaseModel):
    prod_uri: str
    dev_uri: str
    connection_timeout: int
    pool_size: int
    pool_timeout: int
    uri: str | None = None


class API(BaseModel):
    host: str
    port: int


class ML(BaseModel):
    mlflow_tracking_url: str
    scut_data_path: str
    remote_ip: str | None = None


class Auth(BaseModel):
    secret_key: str
    algorithm: str
    access_token_expire_m: int
    salt: bytes
    admin_name: str
    admin_password: str


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env.example", ".env"],
        env_nested_delimiter="__",
    )
    env: Literal["dev", "prod"]
    prod: bool = False
    db: DB
    api: API
    ml: ML
    auth: Auth
    logging: Logging

    def model_post_init(self, context):
        self.prod = self.env == "prod"
        self.logging.level = self.prod and "INFO" or "DEBUG"
        self.logging.config = (
            self.prod and self.logging.prod_config or self.logging.dev_config
        )
        self.logging.file = (
            self.prod
            and self.logging.file
            or self.logging.file.replace(".log", "_dev.log")
        )
        self.db.uri = (
            self.prod
            and self.db.prod_uri.replace("postgresql://", "postgresql+asyncpg://")
            or self.db.dev_uri.replace("sqlite://", "sqlite+aiosqlite://")
        )
        self.db.__delattr__("prod_uri")
        self.db.__delattr__("dev_uri")


config = Config()

warnings.simplefilter(action="ignore", category=FutureWarning)
