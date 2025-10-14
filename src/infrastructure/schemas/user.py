from datetime import datetime

from pydantic import Field

from .base import Base, IDRead
from .celebrity import Celebrity


class Inference(Base):
    id: int
    user_id: str = Field(examples=[123], description="User ID")
    celebrities: list[Celebrity] = Field(description="List of celebrities")
    attractiveness: float = Field(examples=[4.0], description="Attractiveness score")
    date: datetime = Field(examples=[datetime.now()], description="Inference date")
    picture: bytes | None = Field(None, description="Inference picture")


class UserBase(Base):
    login: str = Field(
        examples=["login"], min_length=3, max_length=30, description="User login"
    )


class UserCreate(UserBase):
    password: str = Field(
        examples=["password"], min_length=3, max_length=30, description="User password"
    )


class UserUpdate(UserBase):
    pass


class UserRead(IDRead, UserBase):
    inferences: list[Inference] = Field(description="List of inferences")
