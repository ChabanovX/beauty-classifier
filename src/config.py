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


class Auth(BaseModel):
    secret_key: str
    algorithm: str
    access_token_expire: int
    salt: str


class App(BaseModel):
    dev: bool = False


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env", ".env.example"], env_nested_delimiter="__"
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

warnings.simplefilter(action="ignore", category=FutureWarning)
