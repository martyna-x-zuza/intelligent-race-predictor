import pytest

from race_predictor.exceptions import (
    DataValidationError,
    RacePredictorError,
    UnsupportedExportFormatError,
)


@pytest.mark.parametrize("error", [DataValidationError, UnsupportedExportFormatError])
def test_all_errors_share_one_base_class(error: type[RacePredictorError]) -> None:
    with pytest.raises(RacePredictorError, match="boom"):
        raise error("boom")


def test_base_error_is_a_plain_exception() -> None:
    assert issubclass(RacePredictorError, Exception)
