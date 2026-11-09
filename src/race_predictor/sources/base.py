"""Activity source abstraction and registry.

Each export format gets an adapter (:class:`ActivitySource` subclass) that turns its rows into
canonical :class:`~race_predictor.domain.Activity` objects. Adapters register themselves with the
:func:`register_source` decorator, so :func:`detect_source` can pick the right one for a file
without the caller knowing which formats exist.

:meth:`ActivitySource.read` is a template method: subclasses only say how to read the raw table
(:meth:`~ActivitySource.read_table`) and how to convert one row
(:meth:`~ActivitySource.parse_row`); skipping undated or invalid rows is shared.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, TypeVar

import pandas as pd

from race_predictor.domain import Activity
from race_predictor.exceptions import DataValidationError, UnsupportedExportFormatError

logger = logging.getLogger(__name__)

#: One row of an export, as produced by ``DataFrame.to_dict(orient="records")``.
Row = Mapping[Hashable, Any]

_REGISTRY: dict[str, type[ActivitySource]] = {}

SourceT = TypeVar("SourceT", bound="type[ActivitySource]")


class ActivitySource(ABC):
    """Adapter that reads one export file and yields canonical activities."""

    #: Registry name, set by :func:`register_source`.
    name: ClassVar[str]
    #: Columns that must all be present for a file to be recognised as this format.
    required_columns: ClassVar[frozenset[str]]

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Cheap check, by sniffing the CSV header, whether ``path`` is in this format."""
        return cls.required_columns.issubset(read_header(path))

    def read(self, path: Path, athlete_id: str) -> Iterator[Activity]:
        """Lazily yield the activities in ``path``, all attributed to ``athlete_id``.

        Rows without a parseable date, or that violate a domain invariant (for example a
        negative distance), are skipped and counted in a warning instead of aborting the load.

        Raises:
            UnsupportedExportFormatError: If ``path`` is not in this source's format.
        """
        if not self.can_read(path):
            raise UnsupportedExportFormatError(f"{path.name} is not a {self.name} export")
        skipped = 0
        for row in self.read_table(path).to_dict(orient="records"):
            try:
                activity = self.parse_row(row, athlete_id)
            except DataValidationError:
                activity = None
            if activity is None:
                skipped += 1
                continue
            yield activity
        if skipped:
            logger.warning("%s: skipped %d unreadable row(s)", path.name, skipped)

    @abstractmethod
    def read_table(self, path: Path) -> pd.DataFrame:
        """Read the raw export into a table with one row per activity."""

    @abstractmethod
    def parse_row(self, row: Row, athlete_id: str) -> Activity | None:
        """Convert one raw row; ``None`` if the row has no usable date.

        Raises:
            DataValidationError: If the parsed values violate a domain invariant.
        """


def register_source(name: str) -> Callable[[SourceT], SourceT]:
    """Class decorator registering an :class:`ActivitySource` subclass under ``name``.

    Raises:
        ValueError: If another source is already registered under ``name``.
    """

    def decorator(cls: SourceT) -> SourceT:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"a source named {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_sources() -> list[str]:
    """Names of all registered sources, sorted."""
    return sorted(_REGISTRY)


def get_source(name: str) -> ActivitySource:
    """Instantiate the source registered as ``name``.

    Raises:
        UnsupportedExportFormatError: If no source has that name.
    """
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise UnsupportedExportFormatError(
            f"unknown source {name!r}; available: {available_sources()}"
        ) from None


def detect_source(path: str | Path) -> ActivitySource:
    """Instantiate the registered source that recognises ``path``.

    Raises:
        UnsupportedExportFormatError: If no registered source can read ``path``.
    """
    path = Path(path)
    for cls in _REGISTRY.values():
        if cls.can_read(path):
            return cls()
    raise UnsupportedExportFormatError(
        f"{path.name} is not a supported export (known formats: {available_sources()})"
    )


def read_header(path: Path) -> list[str]:
    """Column names of the CSV file ``path``; empty if it cannot be read as CSV."""
    try:
        return [str(column) for column in pd.read_csv(path, nrows=0).columns]
    except (OSError, UnicodeDecodeError, ValueError):
        return []
