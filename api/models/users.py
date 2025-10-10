from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class UserCreate(BaseModel):
    login: str = Field(..., examples=["johndoe"])
    password: str = Field(..., examples=["P@ssw0rd!"])
    
class UserBase(BaseModel):
    user_id: UUID
    login: str
    is_active: bool
    is_verified: bool
    creation_date: datetime

class UserResponse(BaseModel):
    user_id: UUID
    login: str
    is_active: bool
    is_verified: bool
    creation_date: datetime

class LoginRequest(BaseModel):
    login: str
    password: str

class PhotoCreate(BaseModel):
    user_id: UUID
    storage_uri: str  # dvc://... / s3://...
    sha256: Optional[str] = None

class PhotoResponse(BaseModel):
    photo_id: UUID
    user_id: UUID
    storage_uri: str
    status: str
    created_at: datetime

class InferenceCreate(BaseModel):
    user_id: UUID
    photo_id: UUID
    model_version_id: UUID

class MatchOut(BaseModel):
    celebrity_id: UUID
    name: str
    similarity: float
    rank: int

class InferenceResponse(BaseModel):
    inference_id: UUID
    user_id: UUID
    photo_id: UUID
    model_version_id: UUID
    beauty_score: float
    created_at: datetime
    matches: List[MatchOut]
