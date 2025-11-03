from src.interfaces.api.v1.schemas import Celebrity
from src.infrastructure.repositories import CelebrityRepository
from .crud import CRUDService


class CelebrityService(CRUDService[CelebrityRepository, Celebrity]):
    async def get_picture(self, id: int) -> bytes:
        return await self.repository.get_picture(id)
