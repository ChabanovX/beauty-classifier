# models.py
import uuid
from datetime import datetime
from typing import Optional, List
from enum import Enum as PyEnum
from uuid import UUID
from pydantic import BaseModel, Field
from sqlalchemy import (
    String, Boolean, DateTime, ForeignKey, Float, Integer, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship





class Base(DeclarativeBase):
    pass

# Пользователь без почты/роадмапа
class User(Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    login: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    photos: Mapped[list["Photo"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    inferences: Mapped[list["Inference"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_users_login", "login"),)

# Фото пользователя: только ссылки/мета
class PhotoStatus(str, PyEnum):
    queued = "queued"
    processed = "processed"
    failed = "failed"

class Photo(Base):
    __tablename__ = "photos"
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)

    storage_uri: Mapped[str] = mapped_column(String(512), nullable=False)  # dvc://... или s3://...
    sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[PhotoStatus] = mapped_column(String(16), default=PhotoStatus.queued.value, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="photos")
    inferences: Mapped[list["Inference"]] = relationship(back_populates="photo", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_photos_user_status", "user_id", "status"),
    )

# Версия модели/наборов
class ModelVersion(Base):
    __tablename__ = "model_versions"
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False)  # например "face-enc-v3"
    registry_ref: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)  # ссылка в MLflow/реестре
    celebrity_set_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # версия набора знаменитостей
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (UniqueConstraint("name", "celebrity_set_version", name="uq_model_name_setver"),)

# Инференс: скора и тайминги
class Inference(Base):
    __tablename__ = "inferences"
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    photo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("photos.id", ondelete="CASCADE"), index=True, nullable=False)
    model_version_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_versions.id", ondelete="RESTRICT"), index=True, nullable=False)

    beauty_score: Mapped[float] = mapped_column(Float, nullable=False)  # 0..100 или 0..1
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    user: Mapped["User"] = relationship(back_populates="inferences")
    photo: Mapped[bytes] = mapped_column(nullable=False)
    model_version: Mapped["ModelVersion"] = relationship()

    matches: Mapped[list["Match"]] = relationship(back_populates="inference", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_inferences_user_created", "user_id", "created_at"),
        Index("ix_inferences_photo_model", "photo_id", "model_version_id"),
    )

# Справочник знаменитостей
class Celebrity(Base):
    __tablename__ = "celebrities"
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    ref_images_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # dvc://... набор эталонных фото
    embeddings_uri: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # ссылка на файл с эмбеддингами
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    embedding_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    __table_args__ = (UniqueConstraint("name", name="uq_celebrity_name"),)

# Матчи инференса с селебрити
class Match(Base):
    __tablename__ = "matches"
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    inference_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("inferences.id", ondelete="CASCADE"), index=True, nullable=False)
    celebrity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("celebrities.id", ondelete="RESTRICT"), index=True, nullable=False)

    similarity: Mapped[float] = mapped_column(Float, nullable=False)  # 0..1 косинус/др
    rank: Mapped[int] = mapped_column(Integer, nullable=False)        # 1 — лучший матч
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    inference: Mapped["Inference"] = relationship(back_populates="matches")
    celebrity: Mapped["Celebrity"] = relationship()

    __table_args__ = (
        UniqueConstraint("inference_id", "celebrity_id", name="uq_inference_celebrity"),
        Index("ix_matches_inference_rank", "inference_id", "rank"),
    )

