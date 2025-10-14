from datetime import datetime

from sqlalchemy import ForeignKey, LargeBinary, Table, Column
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class CreatedAtMixin(Base):
    __abstract__ = True
    created_at: Mapped[datetime] = mapped_column(
        nullable=False, server_default=func.now()
    )


# Association table for many-to-many between Inference and Celebrity
association_table = Table(
    "inference_celebrities",
    CreatedAtMixin.metadata,
    Column("inference_id", ForeignKey("inferences.id"), primary_key=True),
    Column("celebrity_id", ForeignKey("celebrities.id"), primary_key=True),
)


class User(CreatedAtMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    login: Mapped[str] = mapped_column(unique=True, nullable=False, index=True)
    password: Mapped[str] = mapped_column(nullable=False)
    inferences: Mapped[list["Inference"]] = relationship(
        back_populates="user", cascade="all, delete", lazy="selectin"
    )


class Celebrity(CreatedAtMixin):
    __tablename__ = "celebrities"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False, index=True)
    picture: Mapped[bytes] = mapped_column(nullable=False, deferred=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=True)
    inferences: Mapped[list["Inference"]] = relationship(
        secondary=association_table, back_populates="celebrities", cascade="all, delete"
    )


class Inference(CreatedAtMixin):
    __tablename__ = "inferences"
    id: Mapped[int] = mapped_column(primary_key=True)
    attractiveness: Mapped[float] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    picture: Mapped[bytes] = mapped_column(nullable=False, deferred=True)
    user: Mapped["User"] = relationship(back_populates="inferences")
    celebrities: Mapped[list["Celebrity"]] = relationship(
        secondary=association_table,
        back_populates="inferences",
        cascade="all, delete",
        lazy="selectin",  # Eagerly loads celebrities when querying Inference
    )
