import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg
import dotenv

dotenv.load_dotenv()

"postgres://user:password@db:5432/db"
"postgres+asyncpg://user:password@db:5432/db"

_pool: asyncpg.Pool = None
    
_db_config = {
    "user": os.getenv("DB_USER", "user"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "database": os.getenv("DB_NAME", "db"),
    "host": os.getenv("DB_HOST", "db"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "min_size": int(os.getenv("DB_POOL_MIN", 5)),
    "max_size": int(os.getenv("DB_POOL_MAX", 20)),
    "timeout": 30,
}

async def connect(max_retries: int = 10, delay: float = 1):
    """Wait for the database to be available, then create
    a connection pool"""
    global _pool
    retries = 0
    while retries < max_retries:
        try:
            test_conn = await asyncpg.connect(
                user=_db_config["user"],
                password=_db_config["password"],
                database=_db_config["database"],
                host=_db_config["host"],
                port=_db_config["port"],
                timeout=5,
            )
            await test_conn.close()
            print("Database is ready.")
            break
        except Exception as e:
            print(f"Waiting for DB... ({retries+1}/{max_retries}) - {e}")
            await asyncio.sleep(delay)
            retries += 1
    else:
        print("Database did not become available in time. Exiting.")
        sys.exit(1)

    # DB is ready, now initialize the connection pool
    _pool = await asyncpg.create_pool(**_db_config)

async def close():
    global _pool
    await _pool.close()

@asynccontextmanager
async def connection():
    global _pool
    """Get a connection from the pool with context manager"""
    if not _pool:
        await connect()

    conn: asyncpg.connection.Connection = await _pool.acquire()
    try:
        yield conn
    # except Exception as e:
    #     conn.rollback()
    finally:
        await _pool.release(conn)