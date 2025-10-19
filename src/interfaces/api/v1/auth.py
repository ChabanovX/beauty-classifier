from fastapi import APIRouter, Depends

from src.application.services import UserService
from src.interfaces.api.exc import (
    InvalidCredentialsHTTPException,
    AlreadyExistsHTTPException,
)
from src.interfaces.api.schemas import UserCreate, Token

auth_router = APIRouter(prefix="/auth", tags=["Auth"])


@auth_router.post("/register")
async def register(
    data: UserCreate,
    service: UserService = Depends(),
) -> Token:
    token = await service.register(data)
    if not token:
        raise AlreadyExistsHTTPException
    return token


@auth_router.post("/login")
async def login(data: UserCreate, service: UserService = Depends()) -> Token:
    token = await service.login(data)
    if not token:
        raise InvalidCredentialsHTTPException
    return token
