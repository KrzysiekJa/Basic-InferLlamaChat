from pathlib import Path

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.logger import init_logging, logger
from app.rate_limiting import limiter
from app.config import settings
from app.api import register_routes


BASE_PATH = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


app: FastAPI = FastAPI(title="Llama4Infer ChatApp")
init_logging()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
register_routes(app)


@app.middleware("http")
async def log_requests(request: Request, call_next: callable):
    logger.debug(f"{request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    logger.debug(f"Completed with status {response.status_code}")
    return response


@app.get("/", status_code=status.HTTP_307_TEMPORARY_REDIRECT)
@limiter.limit("30/minute")
async def root(request: Request):
    return RedirectResponse(url="/ui")


@app.get("/ui", status_code=status.HTTP_200_OK)
@limiter.limit("30/minute")
async def ui(request: Request):
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "minOutTokens": settings.chat.OUTPUT_MIN_TOKENS,
            "maxOutTokens": settings.chat.OUTPUT_MAX_TOKENS,
        },
    )


if "__main__" == __name__:
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="debug"
    )
