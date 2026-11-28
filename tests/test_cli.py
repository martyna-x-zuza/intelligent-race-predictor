import json
from pathlib import Path

import pytest

from race_predictor import __version__
from race_predictor.cli import main

DATA = Path(__file__).parents[1] / "data"


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_a_subcommand_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([])
    assert exit_info.value.code == 2
    assert "baselines" in capsys.readouterr().err


def test_unknown_model_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["baselines", "--models", "Transformer"])
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.skipif(not DATA.exists(), reason="sample exports not present")
def test_baselines_writes_the_report_and_the_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report, artifacts = tmp_path / "RESULTS.md", tmp_path / "baselines"
    argv = ["baselines", "--data", str(DATA), "--report", str(report)]
    main([*argv, "--artifacts", str(artifacts), "--models", "Riegel", "Ridge", "--n-splits", "3"])

    assert "Riegel" in capsys.readouterr().out
    assert "3-fold cross-validation" in report.read_text()
    params = json.loads((artifacts / "params.json").read_text())
    assert params["models"] == ["Riegel", "Ridge"]
    assert params["n_splits"] == 3
    assert params["data_dir"] == str(DATA)
    assert params["n_races"] > 50
    results = (artifacts / "results.csv").read_text().splitlines()
    assert results[0].startswith("model,protocol,mae_time_min")
    assert len(results) == 1 + 2 * 2  # header + 2 models x 2 protocols


@pytest.mark.skipif(not DATA.exists(), reason="sample exports not present")
def test_predict_prints_a_time_and_a_range_per_distance(capsys: pytest.CaptureFixture[str]) -> None:
    export = next((DATA / "strava_activities").glob("*.csv"))
    main(["predict", str(export), "--data", str(DATA), "--distances-km", "5", "10"])
    lines = capsys.readouterr().out.splitlines()
    assert "predicted" in lines[0]
    assert [line.split("km")[0].strip() for line in lines[1:]] == ["5", "10"]
    assert all(" - " in line for line in lines[1:])


def test_predict_reports_an_unsupported_file(tmp_path: Path) -> None:
    export = tmp_path / "other.csv"
    export.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="error:"):
        main(["predict", str(export)])
