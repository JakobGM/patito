"""Module implementing the XDG directory standard."""
import os
from pathlib import Path
from typing import Optional


def cache_home(application: Optional[str] = None) -> Path:
    """
    Return path to directory containing user-specific non-essential data files.

    Args:
        application: An optional name of an application for which to return an
            application-specific cache directory for.

    Returns:
        A path object pointing to a directory to store cache files.
    """
    path = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).resolve()
    if application:
        path = path / application
    path.mkdir(exist_ok=True, parents=True)
    return path
