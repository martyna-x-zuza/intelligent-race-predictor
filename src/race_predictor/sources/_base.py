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

_logger = logging.getLogger(__name__)

#: One row of an export, as produced by ``DataFrame.to_dict(orient="records")``.
Row = Mapping[Hashable, Any]

_REGISTRY: dict[str, type[ActivitySource]] = {}

_SourceT = TypeVar("_SourceT", bound="type[ActivitySource]")


class ActivitySource(ABC):
    """Adapter that reads one export file and yields canonical activities."""

    #: Registry name, set by :func:`register_source`.
    name: ClassVar[str]
    #: Columns that must all be present for a file to be recognised as this format.
    required_columns: ClassVar[frozenset[str]]

    @classmethod
    def can_read(cls, path: Path) -> bool:
        """Check cheaply, by sniffing the CSV header, whether a file is in this format.

        Args:
            path: The file to check.

        Returns:
            ``True`` if the header has all :attr:`required_columns`.
        """
        return cls.required_columns.issubset(_read_header(path))

    def read(self, path: Path, athlete_id: str) -> Iterator[Activity]:
        """Lazily read the activities of one export file.

        Rows without a parseable date, or that violate a domain invariant (for example a
        negative distance), are skipped and counted in a warning instead of aborting the load.

        Args:
            path: The export file.
            athlete_id: Id every activity is attributed to.

        Yields:
            One :class:`Activity` per readable row, in file order.

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
            _logger.warning("%s: skipped %d unreadable row(s)", path.name, skipped)

    @abstractmethod
    def read_table(self, path: Path) -> pd.DataFrame:
        """Read the raw export.

        Args:
            path: The export file.

        Returns:
            The raw table, one row per activity.
        """

    @abstractmethod
    def parse_row(self, row: Row, athlete_id: str) -> Activity | None:
        """Convert one raw row to an activity in canonical units.

        Args:
            row: One row of :meth:`read_table`, as a column-to-value mapping.
            athlete_id: Id the activity is attributed to.

        Returns:
            The activity, or ``None`` if the row has no usable date.

        Raises:
            DataValidationError: If the parsed values violate a domain invariant.
        """


def register_source(name: str) -> Callable[[_SourceT], _SourceT]:
    """Build a class decorator that registers an :class:`ActivitySource` subclass.

    Args:
        name: Name to register the source under; also set as its ``name`` attribute.

    Returns:
        The decorator. It returns the class unchanged apart from ``name``.

    Raises:
        ValueError: When the decorator is applied and another class already has ``name``.
    """

    def decorator(cls: _SourceT) -> _SourceT:
        """Register ``cls`` under the enclosing ``name``.

        Args:
            cls: The adapter class.

        Returns:
            The same class.
        """
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"a source named {name!r} is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def available_sources() -> list[str]:
    """List the registered sources.

    Returns:
        Names of all registered sources, sorted.
    """
    return sorted(_REGISTRY)


def get_source(name: str) -> ActivitySource:
    """Instantiate a source by name.

    Args:
        name: Registry name, for example ``"strava"``.

    Returns:
        A new instance of the source.

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
    """Find the source that can read a file.

    Args:
        path: The export file.

    Returns:
        A new instance of the first registered source that recognises the file.

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


def _read_header(path: Path) -> list[str]:
    """Read the column names of a CSV file.

    Args:
        path: The file to read.

    Returns:
        The column names; empty if the file is missing or cannot be read as CSV.
    """
    try:
        return [str(column) for column in pd.read_csv(path, nrows=0).columns]
    except (OSError, UnicodeDecodeError, ValueError):
        return []
