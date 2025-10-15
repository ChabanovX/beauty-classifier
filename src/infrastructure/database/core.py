from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)

from src.config import config

_engine: AsyncEngine | None = None

_async_session: async_sessionmaker | None = None


@asynccontextmanager
async def db_engine_lifespan():
    global _engine, _async_session
    _engine = create_async_engine(
        config.db.connection_string,
        pool_size=config.db.pool_size,
        pool_timeout=config.db.pool_timeout,
        pool_pre_ping=True,
    )
    _async_session = async_sessionmaker(_engine, expire_on_commit=False)
    yield
    await _engine.dispose()


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with _async_session() as session:
        async with session.begin():
            yield session
