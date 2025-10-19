from typing import override
import logging

from fastapi import Depends

from src.infrastructure.repositories import UserRepository
from src.interfaces.api.schemas import IDMixin, UserCreate, UserRead, Inference
from .crud import CRUDService
from .security import SecurityService

logger = logging.getLogger(__name__)


class UserService(CRUDService[UserRepository, UserRead]):
    def __init__(self, repository: UserRepository = Depends()):
        super().__init__()
        self.repository = repository
        self.read_schema = UserRead

    @override
    async def create(self, data: UserCreate):
        data.password = SecurityService.hash_password(data.password)
        user = await self.repository.create(**data.model_dump())
        if not user:
            return None
        return IDMixin.model_validate(user)

    @override
    async def get(self, id_login: int | str):
        user = await self.repository.get(id_login)
        if not user:
            return None
        return self.read_schema.model_validate(user)

    async def get_inferences(self, id: int):
        return [
            Inference.model_validate(inference)
            for inference in await self.repository.get_inferences(id)
        ]

    async def login(self, data: UserCreate):
        user = await self.repository.get(data.name)

        if not user or not SecurityService.verify_password(
            data.password, user.password
        ):
            return None  # not exists or incorrect password

        return SecurityService.issue_token(
            user_id=user.id, username=user.name, role=user.role
        )

    async def register(self, data: UserCreate):
        user = await self.repository.get(data.name)
        if user:
            return None  # already exists

        data.password = SecurityService.hash_password(data.password)
        user = await self.repository.create(**data.model_dump())
        if not user:
            return None
        return SecurityService.issue_token(
            user_id=user.id, username=user.name, role=user.role
        )
