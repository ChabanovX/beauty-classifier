from pydantic import BaseModel, ConfigDict


class Base(BaseModel):
    model_config = ConfigDict(from_attributes=True, strict=True, extra="forbid")


class IDMixin(BaseModel):
    id: int
