import pytest
from alembic import command as co
from alembic.config import Config as AbConfig

from src.config import config


@pytest.fixture(autouse=True)
def test_db():
    global config
    config.env = "dev"
    config.db.uri = "sqlite+aiosqlite:///testdb.sqlite3"

    ab_config = AbConfig("alembic.ini")
    ab_config.set_main_option("sqlalchemy.url", config.db.uri)

    co.upgrade(ab_config, "head")
    yield
    co.downgrade(ab_config, "base")
