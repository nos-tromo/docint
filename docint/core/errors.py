"""Global error handlers: generic client-visible bodies, full detail to logs."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers that keep exception detail out of response bodies.

    Args:
        app: The FastAPI application to attach the handlers to.
    """

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
        return JSONResponse(status_code=422, content={"detail": "Invalid request."})

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.opt(exception=exc).error(f"Unhandled error on {request.method} {request.url.path}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})
