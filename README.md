# Race Predictor

Predict running race times (5 km to marathon) from Strava and Garmin training data.

> **Status:** work in progress, built milestone by milestone.

## Milestones

- [x] 1. Project setup and data model
- [x] 2. Data ingestion
- [ ] 3. Cleaning and feature pipeline
- [ ] 4. Baselines and evaluation
- [ ] 5. LSTM and GRU models
- [ ] 6. Application and final delivery

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

| Task | Command |
|---|---|
| Run tests (with coverage) | `pytest` |
| Lint | `ruff check .` |
| Check formatting | `ruff format --check .` |
| Type check (strict) | `mypy` |

The same four checks run in GitHub Actions on Python 3.11 and 3.12 for every push to `main` and
every pull request.

## Project structure

```text
src/race_predictor/
├── exceptions.py       # RacePredictorError and its subclasses
├── domain/
│   └── models.py       # Activity, ActivityType, Athlete, Race, RaceCategory
└── sources/
    ├── base.py         # ActivitySource adapter, registry, format detection
    ├── parsing.py      # raw cell -> canonical number helpers
    ├── strava.py       # Strava activities.csv adapter
    ├── garmin.py       # Garmin Connect CSV adapter (Polish headers)
    └── loader.py       # files -> Athlete objects -> DataFrame
tests/                  # mirrors the package layout
```

## Domain model

All quantities use canonical units (metres, seconds, m/s, bpm, °C); a missing measurement is `None`.

- `Activity`: immutable record of one activity (`frozen`, `slots`); rejects negative quantities.
- `Athlete`: an athlete and their activities; guarantees every activity belongs to them and
  exposes `runs()` and an estimated `hr_max`.
- `Race`: a run together with its `RaceCategory` (5k, 10k, half marathon, marathon or other).

The domain layer checks only structural invariants. Plausibility checks on messy real-world data
(impossible speeds or heart rates) are part of the cleaning pipeline (milestone 3).

## Data ingestion

Each export format has an adapter, a subclass of `ActivitySource` that registers itself with
the `@register_source("name")` decorator. `detect_source(path)` sniffs the CSV header and picks
the matching adapter, so adding a new format means writing one class and nothing else.

```python
from race_predictor.sources import athletes_to_frame, iter_athletes, load_athlete

athlete = load_athlete("my_strava_export.csv")  # format detected from the header
athletes = iter_athletes("data/strava_activities")  # lazy: one athlete per file
frame = athletes_to_frame(athletes)  # one row per activity, NaN for missing
```

- `ActivitySource.read` is a template method and a generator: adapters only implement
  `read_table` and `parse_row`, while skipping rows without a date or with values that break a
  domain invariant (logged as a warning) is shared.
- Units are converted to canonical ones in the adapters: Strava distance is in km, Garmin stores
  run pace in min/km, swim pace in min/100 m and ride speed in km/h, with decimal commas and
  `--` placeholders.
- Negative elevation readings are barometric noise and are treated as missing instead of
  dropping the activity.
- Athletes loaded from a directory are named `<source>_<n>`, not after the (possibly personal)
  file names.

## Errors

Every error raised on purpose derives from `RacePredictorError`, so callers can catch one type:

- `DataValidationError`: data violates a domain invariant.
- `UnsupportedExportFormatError`: a file is not a recognised Strava or Garmin export.
