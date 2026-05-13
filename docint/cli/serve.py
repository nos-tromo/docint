"""Console entry point that runs the FastAPI app via uvicorn."""

from __future__ import annotations

import uvicorn

from docint.utils.env_cfg import load_serve_config


def main() -> None:
    """Run the FastAPI app on the configured bind address."""
    cfg = load_serve_config()
    uvicorn.run("docint.core.api:app", host=cfg.host, port=cfg.port, reload=False)


if __name__ == "__main__":
    main()
