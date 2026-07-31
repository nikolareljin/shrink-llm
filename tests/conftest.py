"""Shared test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tomllib is stdlib from 3.11. pyproject declares requires-python = ">=3.10" and CI runs 3.10,
# so the 3.10 leg needs tomli, which the dev extra supplies under a version marker.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on the 3.10 CI leg
    import tomli as tomllib

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The repository root, resolved from this file rather than the working directory."""
    return _REPO_ROOT


@pytest.fixture(scope="session")
def pyproject() -> dict:
    """Parsed pyproject.toml, resolved from the repo root rather than the CWD."""
    return tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
