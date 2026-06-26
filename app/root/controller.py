from fastapi import APIRouter, status, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.rate_limiting import limiter
from app.config import settings, BASE_PATH


TEMPLATES = Jinja2Templates(directory=str(BASE_PATH / "templates"))


router = APIRouter()


@router.get(
    "/", status_code=status.HTTP_307_TEMPORARY_REDIRECT, include_in_schema=False
)
@limiter.limit("30/minute")
async def root(request: Request):
    return RedirectResponse(url="/ui")


@router.get("/ui", status_code=status.HTTP_200_OK, include_in_schema=False)
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
