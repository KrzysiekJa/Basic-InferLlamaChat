from fastapi import FastAPI

from app.inference.controllers import infer_router


def register_routes(app: FastAPI) -> None:
    app.include_router(infer_router)
