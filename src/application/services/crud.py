from abc import ABC
from typing import TypeVar, Generic

from pydantic import BaseModel

from src.infrastructure.repositories import CRUDRepository

RepositoryType = TypeVar("RepositoryType", bound=CRUDRepository)
ReadSchemaType = TypeVar("ReadSchemaType", bound=BaseModel)


class CRUDService(ABC, Generic[RepositoryType, ReadSchemaType]):
    repository: RepositoryType
    read_schema: type[ReadSchemaType]

    async def create(self, **data):
        db_model = await self.repository.create(**data)
        if not db_model:
            return None
        return self.read_schema.model_validate(db_model)

    async def get(self, id: int):
        db_model = await self.repository.get(id)
        if not db_model:
            return None
        return self.read_schema.model_validate(db_model)

    async def find(self, page: int = 1, limit: int = 100):
        offset = (page - 1) * limit
        return [
            self.read_schema.model_validate(user)
            for user in await self.repository.find(offset, limit)
        ]

    async def delete(self, id: int):
        return await self.repository.delete(id)

    async def update(self, id: int, **data):
        return await self.repository.update(id, **data)
