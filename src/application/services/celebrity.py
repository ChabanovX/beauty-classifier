from fastapi import Depends

from src.interfaces.api.v1.schemas import Celebrity
from src.infrastructure.repositories import CelebrityRepository
from .crud import CRUDService


class CelebrityService(CRUDService[CelebrityRepository, Celebrity]):
    read_schema = Celebrity

    def __init__(self, repository: CelebrityRepository = Depends()):
        super().__init__()
        self.repository = repository
        self.read_schema = Celebrity

    async def get_picture(self, id: int) -> bytes:
        return await self.repository.get_picture(id)
