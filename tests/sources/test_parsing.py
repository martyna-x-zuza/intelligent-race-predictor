import math

import pytest

from race_predictor.sources.parsing import parse_duration, scale, to_elevation, to_float


@pytest.mark.parametrize(
    ("value", "expected"),
    [("152", 152.0), ("6,44", 6.44), (" 3.5 ", 3.5), (7, 7.0), (2.5, 2.5)],
)
def test_to_float_parses_numbers(value: object, expected: float) -> None:
    assert to_float(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, math.nan, "", "--", "None", "nan", "abc"])
def test_to_float_returns_none_for_missing_values(value: object) -> None:
    assert to_float(value) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [("01:02:03", 3723.0), ("5:56", 356.0), ("00:00:01.5", 1.5), ("2900", 2900.0)],
)
def test_parse_duration(value: str, expected: float) -> None:
    assert parse_duration(value) == pytest.approx(expected)


@pytest.mark.parametrize("value", [None, math.nan, "--", "1:xx", ""])
def test_parse_duration_returns_none_for_invalid_values(value: object) -> None:
    assert parse_duration(value) is None


def test_scale_keeps_missing_values_missing() -> None:
    assert scale(2.0, 1000) == 2000.0
    assert scale(None, 1000) is None


def test_negative_elevation_is_treated_as_missing() -> None:
    assert to_elevation("12,5") == 12.5
    assert to_elevation("0") == 0.0
    assert to_elevation("-0.4") is None
    assert to_elevation("--") is None
