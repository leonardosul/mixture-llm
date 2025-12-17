"""mixture-llm: Lightweight Mixture of Agents pipeline framework."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from .core import (
    Aggregate,
    Dropout,
    Filter,
    Map,
    Propose,
    Rank,
    Refine,
    Sample,
    Shuffle,
    Synthesize,
    Take,
    Vote,
    run,
)

__all__ = [
    "Shuffle",
    "Dropout",
    "Sample",
    "Take",
    "Filter",
    "Map",
    "Propose",
    "Synthesize",
    "Aggregate",
    "Refine",
    "Rank",
    "Vote",
    "run",
    "__version__",
]

# TODO: Update version fallback using release-please
try:
    __version__ = _version("mixture-llm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"
