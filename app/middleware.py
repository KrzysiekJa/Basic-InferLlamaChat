from fastapi import Request
from app.logger import logger


async def log_requests(request: Request, call_next: callable):
    logger.debug(f"{request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    response = await call_next(request)
    logger.debug(f"Completed with status {response.status_code}")
    return response
