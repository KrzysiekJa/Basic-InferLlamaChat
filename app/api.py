from fastapi import FastAPI

from app.predict.controller import router as predict_router


def register_routes(app: FastAPI) -> None:
    app.include_router(predict_router, prefix="api/v1/predict", tags=["predict", "inference"])
