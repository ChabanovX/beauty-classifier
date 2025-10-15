import platform
import subprocess
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
    access_token_expire: int
    salt: str


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=[".env.example", ".env"],
        env_nested_delimiter="__",
        cli_implicit_flags=True,
        cli_parse_args=True,
    )
    prod: bool = False
    db: DB
    api: API
    ml: ML
    auth: Auth
    logging: Logging

    def model_post_init(self, context):
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


config = Config()

warnings.simplefilter(action="ignore", category=FutureWarning)


def remote_vm_reachable():
    if not config.ml.remote_ip:
        return True
    param = "-c" if platform.system().lower() == "Windows" else "-n"
    command = [
        "ping",
        param,
        "1",
        config.ml.remote_ip,
    ]
    result = subprocess.run(command, capture_output=True, timeout=5).returncode
    return result == 0


if not remote_vm_reachable():
    raise RuntimeError(
        f"Remote VM ({config.ml.remote_ip}) not reachable. Make sure it is running or try turning off VPN"
    )
