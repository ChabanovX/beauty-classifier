from typing import Annotated

from fastapi import Depends

from src.infrastructure.schemas import Celebrity
from src.infrastructure.repositories import CelebrityRepository
from .base import BaseService


class CelebrityService(BaseService[CelebrityRepository]):
    def __init__(self, repository: Annotated[CelebrityRepository, Depends()]):
        self.repository = repository

    async def create(self, data: Celebrity):
        celeb = await self.repository.create(**data.model_dump())
        if not celeb:
            return None
        return Celebrity(id=celeb.id, name=celeb.name, picture=celeb.picture)

    async def update(self, id: int, data: Celebrity):
        celeb = await self.repository.update(id, **data.model_dump())
        if not celeb:
            return None
        return Celebrity(id=celeb.id, name=celeb.name, picture=celeb.picture)

    async def get(self, id: int):
        celeb = await self.repository.get(id)
        if not celeb:
            return None
        return Celebrity(id=celeb.id, name=celeb.name, picture=celeb.picture)

    async def get_picture(self, id: int) -> bytes:
        return await self.repository.get_picture(id)

    async def filter(self, offset: int = 0, limit: int = 100):
        celebs = []
        for celeb in await self.repository.filter(offset, limit):
            celebs.append(
                Celebrity(id=celeb.id, name=celeb.name, picture=celeb.picture)
            )
        return celebs
