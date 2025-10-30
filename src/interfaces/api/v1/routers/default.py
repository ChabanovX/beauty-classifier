from fastapi import APIRouter

default_router = APIRouter(tags=["Default"])


@default_router.get("/health")
async def health():
    return {"status": "ok"}
