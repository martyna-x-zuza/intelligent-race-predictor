from pathlib import Path

import pytest

DATA = Path(__file__).parents[1] / "data"
APP = Path(__file__).parents[1] / "src" / "race_predictor" / "app.py"

testing = pytest.importorskip("streamlit.testing.v1")

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="sample exports not present")


def test_the_page_asks_for_an_export_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACE_PREDICTOR_DATA", str(DATA))
    app = testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert "Upload an export" in app.info[0].value
    assert len(app.metric) == 0


def test_a_sample_athlete_gets_a_prediction_for_every_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RACE_PREDICTOR_DATA", str(DATA))
    app = testing.AppTest.from_file(str(APP), default_timeout=60).run()
    app.sidebar.selectbox[0].select("Sample athlete 1").run()
    assert not app.exception
    assert [metric.label for metric in app.metric] == ["5 km", "10 km", "21.0975 km", "42.195 km"]
    assert all(metric.value.count(":") == 2 for metric in app.metric)
