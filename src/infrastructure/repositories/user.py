from sqlalchemy import select, delete, insert
from sqlalchemy.orm import selectinload

from src.infrastructure.database.models.public import Celebrity

from ..database.models import User, Inference
from .base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    async def get(self, by: str | int) -> User | None:
        """
        Get user

        Args:
            by (str | int): User login or ID

        Returns:
            User | None: User
        """
        if isinstance(by, int):
            column = User.id
        elif isinstance(by, str):
            column = User.login
        query = (
            select(User).options(selectinload(User.inferences)).where(User.column == by)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def filter(self, offset, limit):
        query = (
            select(User)
            .options(selectinload(User.inferences))
            .offset(offset)
            .limit(limit)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_inferences(self, id: int) -> list[Inference]:
        query = (
            select(Inference)
            .where(Inference.user_id == id)
            .options(
                selectinload(Inference.celebrities), selectinload(Inference.picture)
            )
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_photo(self, id: int) -> bytes:
        query = (
            select(Inference)
            .where(Inference.user_id == id)
            .order_by(Inference.date.desc())
            .limit(1)
        )
        result = await self.db.execute(query)
        last_inference = result.scalar_one_or_none()
        if last_inference:
            return last_inference.picture
        return None

    async def create_inference(
        self,
        id: int,
        photo: bytes,
        attractiveness: float,
        celebrity_names: list[str],
    ) -> Inference | None:
        query = select(Celebrity).where(Celebrity.name.in_(celebrity_names))
        result = await self.db.execute(query)
        celebrities = result.scalars().all()
        query = (
            insert(Inference)
            .values(
                user_id=id,
                photo=photo,
                attractiveness=attractiveness,
                celebrities=celebrities,
            )
            .returning(Inference)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def delete_inference(self, id: int) -> bool:
        query = delete(Inference).where(Inference.id == id)
        result = await self.db.execute(query)
        return result.rowcount > 0
