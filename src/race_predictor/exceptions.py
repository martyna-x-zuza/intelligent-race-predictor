"""Exceptions raised by the package.

Everything derives from :class:`RacePredictorError`, so callers (the CLI, the web app) can catch
one type to handle any expected failure and let genuine bugs propagate.
"""


class RacePredictorError(Exception):
    """Base class for all errors raised on purpose by this package."""


class DataValidationError(RacePredictorError):
    """Data violates a domain invariant or is missing something required."""


class UnsupportedExportFormatError(RacePredictorError):
    """A file is not a recognised Strava or Garmin export."""
