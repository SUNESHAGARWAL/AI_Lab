from typing import Annotated

from fastapi import APIRouter, Depends

from api.config import Settings, get_settings

router = APIRouter()


@router.get("/health")
def health(settings: Annotated[Settings, Depends(get_settings)]) -> dict[str, str]:
    return {"status": "ok", "env": settings.app_env}
