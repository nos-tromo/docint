"""Global error handlers: generic client-visible bodies, full detail to logs."""

from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger


def redact_validation_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    """Strip submitted values out of pydantic validation errors.

    A pydantic error carries the offending ``input`` alongside its location
    and type. On this API that input is user content — a malformed
    ``POST /query`` body puts the question itself in the log, which is the
    one path by which query text could ever reach it. The location and the
    error type are what an operator needs; the value is not.

    Args:
        errors (Sequence[Any]): ``RequestValidationError.errors()`` output.

    Returns:
        list[dict[str, Any]]: The same errors with ``input`` replaced by a
        marker and ``ctx`` dropped (it can embed the value too).
    """
    redacted: list[dict[str, Any]] = []
    for error in errors:
        if not isinstance(error, dict):
            redacted.append({"type": "unknown"})
            continue
        safe = {key: value for key, value in error.items() if key not in {"input", "ctx"}}
        if "input" in error:
            safe["input"] = "<redacted>"
        redacted.append(safe)
    return redacted


def install_error_handlers(app: FastAPI) -> None:
    """Register handlers that keep exception detail out of response bodies.

    Args:
        app: The FastAPI application to attach the handlers to.
    """

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning(
            "Validation error on {} {}: {}",
            request.method,
            request.url.path,
            redact_validation_errors(exc.errors()),
        )
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
#:   embedding_unavailable — chat stream: the dense-embedding endpoint could
#:                        not embed the query, so retrieval never ran (a
#:                        configuration/connectivity fault, not a model one)
#:   generation_failed  — chat stream: any other generation failure
#:   summary_failed     — summary stream failure
#:   ingestion_failed   — ingestion finalize-stage failure
#:   save_failed        — uploaded file could not be written (event carries
#:                        the echoed client filename as a structured field)
SSE_ERROR_CODES = frozenset(
    {
        "context_overflow",
        "embedding_unavailable",
        "generation_failed",
        "summary_failed",
        "ingestion_failed",
        "save_failed",
    }
)
