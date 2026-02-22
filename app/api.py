from fastapi import FastAPI

from app.predict.controller import predict_router


def register_routes(app: FastAPI) -> None:
    app.include_router(predict_router)
