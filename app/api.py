from fastapi import FastAPI

from inference.controller import infer_router


def register_routes(app: FastAPI) -> None:
    app.include_router(infer_router)
