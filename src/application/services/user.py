from typing import Annotated

from fastapi import Depends

from src.infrastructure.schemas import (
    IDRead,
    UserCreate,
    UserRead,
    UserUpdate,
    Inference,
)
from src.infrastructure.repositories import UserRepository
from .base import BaseService
from .security import SecurityService
from ..exceptions import (
    NotFoundException,
    AlreadyExistsException,
    IncorrectPasswordException,
)


class UserService(BaseService[UserRepository]):
    def __init__(self, repository: Annotated[UserRepository, Depends()]):
        self.repository = repository

    async def create(self, data: UserCreate) -> IDRead:
        data.password = SecurityService.hash_password(data.password)
        return IDRead.model_validate(await self.repository.create(**data.model_dump()))

    async def update(self, id: int, data: UserUpdate) -> UserRead:
        user = await self.repository.update(id, **data.model_dump())
        if not user:
            return None
        return UserRead(id=user.id, login=user.login, inferences=user.inferences)

    async def get(self, id: int) -> UserRead:
        user = await self.repository.get(id)
        if not user:
            return None
        return UserRead(id=user.id, login=user.login, inferences=user.inferences)

    async def filter(self, page: int = 1, limit: int = 100) -> list[UserRead]:
        offset = (page - 1) * limit
        users = []
        for user in await self.repository.filter(offset, limit):
            users.append(
                UserRead(id=user.id, login=user.login, inferences=user.inferences)
            )
        return users

    async def get_inferences(self, id: int) -> list[Inference]:
        inferences = []
        for inference in await self.repository.get_inferences(id):
            inferences.append(
                Inference(
                    id=inference.id,
                    user_id=inference.user_id,
                    celebrities=inference.celebrities,
                    attractiveness=inference.attractiveness,
                    date=inference.created_at,
                    picture=inference.picture,
                )
            )
        return inferences

    async def login(self, data: UserCreate) -> int:
        user = await self.repository.get(data.login)
        if not user:
            raise NotFoundException
        if not SecurityService.verify_password(data.password, user.password):
            raise IncorrectPasswordException
        return user.id

    async def register(self, data: UserCreate) -> int:
        user = await self.repository.get(data.login)
        if user:
            raise AlreadyExistsException
        data.password = SecurityService.hash_password(data.password)
        user = await self.repository.create(**data.model_dump())
        return user.id
