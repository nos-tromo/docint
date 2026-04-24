"""Shared pytest configuration and fixtures for the docint test suite."""

import sys
import types


class _MagicModule(types.ModuleType):
    """Magic module for handling file types."""

    class Magic:
        """Magic class for handling file types."""

        def __init__(self, mime: bool = True) -> None:
            """Initialize the Magic class.

            Args:
                mime (bool, optional): Whether to use MIME types. Defaults to True.
            """
            self.mime = mime

        def from_file(self, path: str) -> str:
            """Get the MIME type of a file.

            Args:
                path (str): The path to the file.

            Returns:
                str: The MIME type of the file.
            """
            return "application/octet-stream"


def _install_magic_stub() -> None:
    """Install a stub for the magic module."""
    sys.modules.setdefault("magic", _MagicModule("magic"))


def pytest_configure() -> None:
    """Configure pytest by installing necessary stubs."""
    _install_magic_stub()
