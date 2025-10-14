from typing import TypeVar, Generic


from src.infrastructure.repositories import BaseRepository

RepositoryType = TypeVar("RepositoryType", bound=BaseRepository)


class BaseService(Generic[RepositoryType]):
    repository: RepositoryType

    async def delete(self, id: int):
        return await self.repository.delete(id)
