from fastapi import FastAPI

from app.predict.controller import router as predict_router
from app.root.controller import router as root_router


def register_routes(app: FastAPI) -> None:
    app.include_router(root_router, prefix="", tags=["root", "ui"])
    app.include_router(
        predict_router, prefix="/api/v1/predict", tags=["predict", "inference"]
    )
