from abc import ABC
from typing import Generic, TypeVar

from sqlalchemy import select, delete, update, insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from src.infrastructure.database.core import get_session
from src.infrastructure.database.models.base import IDMixin

ModelType = TypeVar("Model", bound=IDMixin)


class CRUDRepository(ABC, Generic[ModelType]):
    model: type[ModelType]

    def __init__(self, db: AsyncSession = Depends(get_session)):
        self.db = db

    async def get(self, id: int):
        query = select(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def find(self, offset: int, limit: int):
        query = select(self.model).offset(offset).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def create(self, **data):
        query = insert(self.model).values(data).returning(self.model)
        try:
            result = await self.db.execute(query)
        except IntegrityError:
            self.db.rollback()
            return None
        return result.scalar_one_or_none()

    async def update(self, id: int, **data):
        query = update(self.model).where(self.model.id == id).values(data)
        try:
            result = await self.db.execute(query)
        except IntegrityError:
            self.db.rollback()
            return None
        return result.rowcount > 0

    async def delete(self, id: int):
        query = delete(self.model).where(self.model.id == id)
        try:
            result = await self.db.execute(query)
        except IntegrityError:
            self.db.rollback()
            return None
        return result.rowcount > 0
