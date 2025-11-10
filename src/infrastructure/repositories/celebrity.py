from sqlalchemy import select

from .crud import CRUDRepository
from ..database.models import Celebrity


class CelebrityRepository(CRUDRepository[Celebrity]):
    async def get_picture(self, id: int) -> bytes:
        query = select(Celebrity.picture).where(Celebrity.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
