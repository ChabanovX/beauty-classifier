import asyncio
import os
import sys
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
import dotenv

dotenv.load_dotenv()

DATABASE_URL = f"postgresql+asyncpg://{os.getenv('DB_USER', 'user')}:" \
               f"{os.getenv('DB_PASSWORD', 'password')}@" \
               f"{os.getenv('DB_HOST', 'db')}:" \
               f"{os.getenv('DB_PORT', '5432')}/" \
               f"{os.getenv('DB_NAME', 'db')}"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=int(os.getenv("DB_POOL_MAX", 20)),
    max_overflow=0,
    pool_timeout=30,
    future=True
)
async_session = async_sessionmaker(engine, expire_on_commit=False)

async def connect(max_retries: int = 10, delay: float = 1):
    """Wait for the database to be available, then test connection"""
    retries = 0
    while retries < max_retries:
        try:
            async with engine.begin() as conn:
                await conn.run_sync(lambda c: None)  # просто тестовый connect
            print("Database is ready.")
            break
        except Exception as e:
            print(f"Waiting for DB... ({retries+1}/{max_retries}) - {e}")
            await asyncio.sleep(delay)
            retries += 1
    else:
        print("Database did not become available in time. Exiting.")
        sys.exit(1)

async def close():
    await engine.dispose()

async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
