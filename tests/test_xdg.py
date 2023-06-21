"""Tests for the patito.xdg module."""
import os
from pathlib import Path

from patito import xdg


def test_xdg_cache_home(monkeypatch, tmpdir):
    """It should yield the correct cache directory according to the standard."""
    xdg_cache_home = tmpdir / ".cache"
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)

    assert xdg.cache_home() == xdg_cache_home
    assert xdg_cache_home.isdir()

    assert xdg.cache_home(application="patito") == xdg_cache_home / "patito"
    assert (xdg_cache_home / "patito").isdir()

    del os.environ["XDG_CACHE_HOME"]
    assert xdg.cache_home() == Path("~/.cache").resolve()
    assert xdg.cache_home(application="patito") == Path("~/.cache/patito").resolve()
