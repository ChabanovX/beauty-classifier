from pydantic import Field

from .base import IDMixin


class Celebrity(IDMixin):
    name: str = Field("name", min_length=3, max_length=30, description="Celebrity name")
    picture: bytes | None = Field(None, description="Celebrity picture")
