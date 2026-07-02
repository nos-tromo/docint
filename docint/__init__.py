"""Document Intelligence: a RAG stack with FastAPI backend and React SPA frontend."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("docint")
except PackageNotFoundError:  # running from source without an installed dist
    __version__ = "0+unknown"
