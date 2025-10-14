from typing import Generic, TypeVar, Annotated

from sqlalchemy import select, delete, update, insert
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from ..database.core import get_session

ModelType = TypeVar("Model", bound=DeclarativeBase)


class BaseRepository(Generic[ModelType]):
    model: type[ModelType]

    def __init__(self, db: Annotated[AsyncSession, Depends(get_session)]):
        self.db = db

    async def get(self, id: int):
        query = select(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def filter(self, offset: int, limit: int):
        query = select(self.model).offset(offset).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def create(self, **data):
        query = insert(self.model).values(data).returning(self.model)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def update(self, id: int, **data):
        query = (
            update(self.model)
            .where(self.model.id == id)
            .values(data)
            .returning(self.model)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def delete(self, id: int):
        query = delete(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        return result.rowcount > 0
