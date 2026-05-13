"""Console entry point that runs the FastAPI app via uvicorn.

Replaces the old Streamlit ``docint.app:run`` entry point.
"""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    """Run the FastAPI app on host 0.0.0.0 with the configured port."""
    host = os.getenv("DOCINT_HOST", "0.0.0.0")
    port = int(os.getenv("DOCINT_PORT", "8000"))
    uvicorn.run("docint.core.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
