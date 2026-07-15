from fastapi import APIRouter

from app.config import settings

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "app_name": settings.app_name}
