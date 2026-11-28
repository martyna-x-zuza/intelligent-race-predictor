"""Streamlit app: upload a Strava or Garmin export and get predicted race times.

Run it with ``just app`` (``streamlit run src/race_predictor/app.py``); it needs the ``app`` extra.
The model is trained on the exports in ``data/`` (or ``$RACE_PREDICTOR_DATA``).
"""

import os
import tempfile
from importlib.util import find_spec
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from race_predictor.evaluation import AVAILABLE_FACTORIES, DEFAULT_FACTORIES, ModelFactory
from race_predictor.exceptions import RacePredictorError
from race_predictor.predictor import RacePredictor, format_time, load_activities

DATA_DIR = Path(os.environ.get("RACE_PREDICTOR_DATA", "data"))

MODELS: dict[str, ModelFactory] = {
    name: make
    for name, make in AVAILABLE_FACTORIES.items()
    if name in DEFAULT_FACTORIES or find_spec("keras")
}


@st.cache_resource(show_spinner="Training the model on the bundled races...")
def get_predictor(model: str) -> RacePredictor:
    """Train a predictor once per server process and model.

    Args:
        model: Name of a model in :data:`MODELS`.

    Returns:
        The fitted predictor.
    """
    return RacePredictor.from_data_dir(DATA_DIR, MODELS[model])


def read_upload(upload: UploadedFile) -> pd.DataFrame:
    """Parse an uploaded export.

    Args:
        upload: The uploaded CSV file.

    Returns:
        The athlete's activity table.
    """
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "export.csv"
        path.write_bytes(upload.getvalue())
        return load_activities(path)


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="Race Predictor")
    st.title("Race Predictor")
    st.caption("Predicts 5 km to marathon times from your training history.")

    samples = {
        f"Sample athlete {number}": path
        for number, path in enumerate(sorted(DATA_DIR.glob("*_activities/*.csv")), start=1)
    }
    with st.sidebar:
        upload = st.file_uploader("Strava activities.csv or Garmin Connect CSV", type="csv")
        sample = st.selectbox("...or try a sample athlete", ["-", *samples])
        model = st.selectbox("Model", list(MODELS))
        elevation_m = st.number_input("Race elevation gain (m)", 0, 3_000, 0, step=50)

    if upload is None and sample == "-":
        st.info("Upload an export, or pick a sample athlete, in the sidebar.")
        return
    try:
        activities = read_upload(upload) if upload else load_activities(samples[str(sample)])
        predictions = get_predictor(str(model)).predict(
            activities, elevation_gain_m=float(elevation_m)
        )
    except RacePredictorError as error:
        st.error(str(error))
        return

    st.subheader("Predicted times if you raced tomorrow")
    for column, prediction in zip(st.columns(len(predictions)), predictions, strict=True):
        column.metric(f"{prediction.distance_m / 1_000.0:g} km", format_time(prediction.time_s))
        column.caption(f"{format_time(prediction.low_s)} - {format_time(prediction.high_s)}")
    st.caption(
        "The range is the 10-90 % error of the model on athletes it was not trained on: "
        "it says how far off a prediction for a new runner typically is."
    )


if __name__ == "__main__":
    main()
