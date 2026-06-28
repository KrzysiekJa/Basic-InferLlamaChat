from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.logger import init_logging, logger
from app.rate_limiting import limiter
from app.config import settings, BASE_PATH
from app.api import register_routes
from app.middleware import log_requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Initialize provider-specific long-lived clients (google-genai) to avoid
    # closing the underlying httpx client before streaming completes.
    if settings.llm.DEFAULT_PROVIDER == "google":
        try:
            from google import genai

            gclient = genai.Client(api_key=settings.llm.api_key)
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
app.mount("/static", StaticFiles(directory=str(BASE_PATH / "static")), name="static")
register_routes(app)

# Add CORS middleware FIRST (executes last in chain due to LIFO order)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.rest.ORIGINS,
    allow_credentials=settings.rest.ALLOWED_CREDENTIALS,
    allow_methods=settings.rest.METHODS,
    allow_headers=settings.rest.HEADERS,
)
app.middleware("http")(log_requests)


if "__main__" == __name__:
    import uvicorn

    uvicorn.run(
        settings.uvicorn.APP_PATH,
        host=settings.uvicorn.IP,
        port=settings.uvicorn.PORT,
        reload=settings.uvicorn.RELOAD,
        log_level=settings.uvicorn.LOG_LEVEL,
    )
