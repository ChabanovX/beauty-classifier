from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Depends, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import ValidationError

from src.application.exceptions import (
    AlreadyExistsException,
    IncorrectPasswordException,
    NotFoundException,
)
from src.application.services import UserService
from src.infrastructure.schemas import UserCreate

user_router = APIRouter(prefix="/users", tags=["User"])

ouath2_scheme = OAuth2PasswordBearer(tokenUrl="/login")


@user_router.post("/")
async def create_user(user: UserCreate, service: Annotated[UserService, Depends()]):
    return await service.create(user)


@user_router.get("/")
async def get(
    service: Annotated[UserService, Depends()],
    page: int = Query(1, description="Page number"),
    limit: int = Query(100, description="Items per page"),
):
    return await service.filter(page, limit)


@user_router.get("/{id}")
async def get_by_id(id: int | str, service: Annotated[UserService, Depends()]):
    return await service.get(id)


# @user_router.post("/{id}/inferences")
# async def create_inference(
#     id: int,
#     ml_service: Annotated[MLService, Depends()],
#     photo: UploadFile,
#     attractiveness: float
# ):
#     if not photo.content_type.startswith("image/"):
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST, detail="File must be an image (JPEG, PNG, JPG, WEBP)"
#         )
#     image_bytes = await photo.read()
#     attractiveness, looak_a_likes = await ml_service.create_inference(id, image_bytes)
#     return inference


@user_router.post("/register")
async def register(user: UserCreate, service: Annotated[UserService, Depends()]):
    print(user)
    try:
        return await service.register(user)
    except AlreadyExistsException:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="User already exists"
        )


@user_router.post("/login")
async def login(
    service: Annotated[UserService, Depends()],
    form_data: Annotated[OAuth2PasswordRequestForm, Depends(ouath2_scheme)],
):
    try:
        return await service.login(
            UserCreate(login=form_data.username, password=form_data.password)
        )
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid login or password format",
        )
    except (NotFoundException, IncorrectPasswordException):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid login or password"
        )
