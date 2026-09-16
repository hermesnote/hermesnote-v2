from fastapi import APIRouter

from indicators.registry import build_registry

router = APIRouter(prefix="/api/indicators", tags=["indicators"])


@router.get("")
def list_indicators():
    return {"groups": build_registry()}
