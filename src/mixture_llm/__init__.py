"""mixture-llm: Lightweight Mixture of Agents pipeline framework."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

from .core import (
    Shuffle,
    Dropout,
    Sample,
    Take,
    Filter,
    Map,
    Propose,
    Synthesize,
    Aggregate,
    Refine,
    Rank,
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

try:
    __version__ = _version("mixture-llm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
