from typing import ClassVar, Literal, Union
import warnings

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logconf import Logging


class DB(BaseModel):
    url: str
    dev_url: str
    connection_timeout: int
    pool_size: int
    pool_timeout: int
    connection_string: str = "unset"


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
        self.logging.level = "INFO" if self.prod else "DEBUG"
        self.logging.config = (
            self.logging.prod_config if self.prod else self.logging.dev_config
        )
        self.logging.file = (
            self.prod
            and self.logging.file
            or self.logging.file.replace(".log", "_dev.log")
        )
        self.db.connection_string = (
            self.prod
            and self.db.url.replace("postgresql://", "postgresql+asyncpg://")
            or self.db.dev_url.replace("sqlite://", "sqlite+aiosqlite://")
        )

    _instance: ClassVar[Union["Config", None]] = None

    @classmethod
    def get(cls):
        if cls._instance:
            return cls._instance
        cls._instance = cls()
        return cls._instance


config = Config.get()

warnings.simplefilter(action="ignore", category=FutureWarning)
