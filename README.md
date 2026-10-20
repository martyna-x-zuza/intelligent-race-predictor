# Race Predictor

Predict running race times (5 km to marathon) from Strava and Garmin training data.

> **Status:** work in progress, built milestone by milestone.

## Milestones

- [x] 1. Project setup and data model
- [ ] 2. Data ingestion
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
└── domain/
    └── models.py       # Activity, ActivityType, Athlete, Race, RaceCategory
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

## Errors

Every error raised on purpose derives from `RacePredictorError`, so callers can catch one type:

- `DataValidationError`: data violates a domain invariant.
- `UnsupportedExportFormatError`: a file is not a recognised Strava or Garmin export.
