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


#: Machine-readable SSE error codes — a closed enum shared with the SPA.
#: Codes are protocol (English tokens, never prose, never exception-derived);
#: the SPA maps them to localized copy and may show the bare token for
#: support triage. Extend deliberately, one code per distinguishable failure:
#:   context_overflow   — chat stream: prompt + retrieval exceed the model window
#:   generation_failed  — chat stream: any other generation failure
#:   summary_failed     — summary stream failure
#:   ingestion_failed   — ingestion finalize-stage failure
#:   save_failed        — uploaded file could not be written (event carries
#:                        the echoed client filename as a structured field)
SSE_ERROR_CODES = frozenset(
    {
        "context_overflow",
        "generation_failed",
        "summary_failed",
        "ingestion_failed",
        "save_failed",
    }
)
