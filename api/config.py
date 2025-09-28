import logging

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    scut_data_path: str = "datasets/scut"


config = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)