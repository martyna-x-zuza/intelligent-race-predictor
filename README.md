# Race Predictor

Predict running race times (5 km to marathon) from Strava and Garmin training data.

> **Status:** work in progress, built milestone by milestone.

## Milestones

- [x] 1. Project setup and data model
- [x] 2. Data ingestion
- [x] 3. Cleaning and feature pipeline
- [x] 4. Baselines and evaluation
- [ ] 5. LSTM and GRU models
- [ ] 6. Application and final delivery

## Development

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The [`justfile`](justfile) is the entry point of the repository: every task is a recipe, and
[`just`](https://github.com/casey/just) is installed with the dev dependencies. Run `just` to list
them.

| Recipe | What it does |
|---|---|
| `just check` | everything CI runs: `lint`, `typecheck` and `test` |
| `just lint` | `ruff check` and `ruff format --check` |
| `just typecheck` | `mypy` in strict mode over `src/` and `tests/` |
| `just test` | `pytest` with coverage; extra arguments go to pytest (`just test -k sources`) |
| `just format` | reformat and apply automatic lint fixes |
| `just baselines` | evaluate the baselines (see below) |
| `just clean` | delete `.artifacts/` |

GitHub Actions runs the same recipes on Python 3.11 and 3.12 for every push to `main` and every
pull request, then `just baselines` and uploads the coverage report and the baseline results as
a build artifact.

**Artifacts.** Every tool writes its caches and outputs to its own directory under the
git-ignored `.artifacts/`, configured in `pyproject.toml`, so the repository root stays clean:

```text
.artifacts/
├── ruff/        lint cache
├── mypy/        type-check cache
├── pytest/      test cache
├── coverage/    coverage data, HTML and XML reports
└── baselines/   params.json (the run's parameters) and results.csv (raw metrics)
```

**Code conventions.**

- Each package's `__init__.py` is its public interface. Modules inside a package are private
  (`_base.py`, `_dataset.py`, ...): code outside the package imports from the package, for example
  `from race_predictor.features import RaceDataset`.
- Inside a module, names used only there start with an underscore (`_read_header`,
  `_MIN_REFERENCE_RUN_M`), and so do instance attributes holding internal state or settings.
- Every function, public or private, has a Google-style docstring describing its parameters
  (`Args:`) and result (`Returns:` / `Yields:`). ruff checks the public ones and
  `tests/test_docstrings.py` checks all of them.

## Command line

The package installs one command, `race-predictor` (also `python -m race_predictor`), with one
subcommand per action. The `just` recipes call it.

```bash
race-predictor baselines --data data --models Riegel Ridge --n-splits 5
race-predictor --help
```

## Project structure

```text
justfile                 # entry point: just check, just baselines, ...
src/race_predictor/
├── cli.py               # the race-predictor command
├── exceptions.py        # RacePredictorError and its subclasses
├── utils.py             # timer() context manager
├── domain/
│   └── _models.py       # Activity, ActivityType, Athlete, Race, RaceCategory
├── sources/
│   ├── _base.py         # ActivitySource adapter, registry, format detection
│   ├── _parsing.py      # raw cell -> canonical number helpers
│   ├── _strava.py       # Strava activities.csv adapter
│   ├── _garmin.py       # Garmin Connect CSV adapter (Polish headers)
│   └── _loader.py       # files -> Athlete objects -> DataFrame
├── processing/
│   ├── _pipeline.py     # Step, Pipeline and the cleaning steps
│   ├── _races.py        # race detection strategies
│   ├── _cleaning.py     # cleaning_pipeline("train" | "predict")
│   └── _windows.py      # leak-free training windows (generator)
├── features/
│   └── _dataset.py      # RaceDataset: one example per race
├── models/
│   ├── _base.py         # RaceModel interface
│   └── _baselines.py    # Riegel formula, ridge regression, gradient boosting
└── evaluation/
    ├── _metrics.py      # MAE, RMSE, MAPE, median APE
    ├── _splits.py       # chronological split and athlete-grouped CV
    └── _runner.py       # model comparison, report and artifacts
tests/                   # mirrors the package layout
data/                    # Strava and Garmin sample exports from the thesis
docs/                    # result reports
```

## Domain model

All quantities use canonical units (metres, seconds, m/s, bpm, °C); a missing measurement is `None`.

- `Activity`: immutable record of one activity (`frozen`, `slots`); rejects negative quantities.
- `Athlete`: an athlete and their activities; guarantees every activity belongs to them and
  exposes `runs()` and an estimated `hr_max`.
- `Race`: a run together with its `RaceCategory` (5k, 10k, half marathon, marathon or other).

The domain layer checks only structural invariants. Plausibility checks on messy real-world data
(impossible speeds or heart rates) are part of the cleaning pipeline.

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

## Cleaning and features

Cleaning is a `Pipeline` of small `Step` objects. Each one takes the activity table and returns a
new one, so steps can be tested alone, reordered or switched off.

```python
from race_predictor.features import build_race_dataset
from race_predictor.processing import cleaning_pipeline

cleaned = cleaning_pipeline("train").run(frame)
dataset = build_race_dataset(cleaned)  # one example per race
```

- **One definition for training and prediction.** `cleaning_pipeline("train")` and
  `cleaning_pipeline("predict")` share the same first steps, so a data fix cannot reach only
  one of them. They differ only where they must: prediction keeps runs without heart rate
  and imputes gaps after race detection, so imputed values never make a run look like a race.
- **Steps:** remove implausible runs (thesis thresholds), rides logged as runs (more than
  1.7x the athlete's median speed), heart-rate sensor spikes and athletes without heart rate.
  Then estimate HRmax, mark runs right after a ride, and detect races.
- **Race detection is a strategy.** Anything with `detect(df) -> Series[bool]` satisfies the
  `RaceDetector` protocol. `HeuristicRaceDetector` ports the thesis heuristic (high heart rate,
  speed spike, standard distance, ...) and `reasons()` shows why a run was flagged.
  `StandardDistanceDetector` is a simple alternative.
- **No leakage.** A race's features use only activities strictly before the race date:
  `training_windows()` and `build_race_dataset()` both enforce it.
- `RaceDataset` holds a run sequence per race (for recurrent models), tabular features
  (cross-training totals, volume and pace summaries, the Riegel reference run) and the race
  pace in s/km as the target.

## Baselines and evaluation

Every model implements `RaceModel` (`fit(dataset)` and `predict(dataset)` returning pace in
s/km), so a formula, a linear model and a neural network are evaluated the same way.

- `RiegelBaseline`: Riegel's formula `T2 = T1 * (D2 / D1) ** 1.06`, applied to the athlete's
  fastest run of at least 5 km before the race.
- `RidgeBaseline`, `GradientBoostingBaseline`: scikit-learn models on the tabular features.
  They share the `TabularModel` base class, so a new model only has to say which estimator
  it uses.

Two protocols, because they answer different questions:

- **chronological** (the thesis protocol): the latest 20 % of each athlete's races are held
  out. Can the model predict an athlete's *next* race?
- **grouped-CV**: 5-fold cross-validation that holds out whole athletes (`GroupKFold`). Does
  it work for a runner it has never seen?

```bash
just baselines
```

This writes the report [`docs/RESULTS_BASELINES.md`](docs/RESULTS_BASELINES.md) and saves the
run's parameters (`params.json`) and raw metrics (`results.csv`) to `.artifacts/baselines/`. The
numbers are regenerated on every run, but the hand-written conclusions are kept. In short: Riegel's formula
is hard to beat. Ridge regression is better on athletes seen in training but much worse on new
ones, and gradient boosting overfits the 160 races.

## Errors

Every error raised on purpose derives from `RacePredictorError`, so callers can catch one type:

- `DataValidationError`: data violates a domain invariant.
- `UnsupportedExportFormatError`: a file is not a recognised Strava or Garmin export.
