from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.logger import init_logging, logger
from app.rate_limiting import limiter
from app.config import settings, BASE_PATH
from app.api import register_routes


TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Initialize provider-specific long-lived clients (google-genai) to avoid
    # closing the underlying httpx client before streaming completes.
    if settings.llm.DEFAULT_PROVIDER == "google":
        try:
            from google import genai

            gclient = genai.Client(api_key=settings.llm.API_KEY)
            # async interface
            app.state.google_genai_client = gclient
            app.state.google_genai_async = getattr(gclient, "aio", None) or gclient
            logger.info("Initialized persistent Google GenAI client on startup.")
        except Exception as exc:  # pragma: no cover - environment specific
            logger.exception("Failed to initialize Google GenAI client: %s", exc)

    yield

    # Shutdown
    # Ensure we close any long-lived google-genai async client gracefully.
    if getattr(app.state, "google_genai_async", None) is not None:
        try:
            aclient = app.state.google_genai_async
            aclose = getattr(aclient, "aclose", None)

            if aclose:
                await aclose()

            # try underlying httpx async client if present
            underlying = getattr(aclient, "_async_httpx_client", None)

            if underlying is not None:
                aclose_under = getattr(underlying, "aclose", None)
                if aclose_under:
                    await aclose_under()
            logger.info("Closed persistent Google GenAI client on shutdown.")
        except Exception:
            logger.exception("Error while closing Google GenAI client")

    limiter.try_acquire = lambda *args, **kwargs: True


app: FastAPI = FastAPI(title="Llama4Infer ChatApp", lifespan=lifespan)
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
