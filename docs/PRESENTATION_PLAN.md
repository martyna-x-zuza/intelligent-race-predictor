# Presentation plan: "Advanced Python in a small ML project" (20 min, code only)

**Audience:** other teams. They write Python daily, many do data/ML work, not all of them use
typing, dataclasses or abstract classes.
**Format:** no slides. You open real files in the editor and walk through them.
**Core message:** *a small ML project stays maintainable when the data flow is made of small,
typed, replaceable pieces.* Every section shows one technique, where it saved us trouble, and
(twice) what we deliberately removed.

## Timing at a glance

| # | Section | Min | Files you open |
|---|---|---|---|
| 0 | Opening: the project and its map | 1.5 | `README.md` structure block |
| 1 | Objects instead of dicts | 3 | `domain/_models.py`, `features/_dataset.py` |
| 2 | Extensible ingestion: ABC, template method, generators | 3.5 | `sources/_base.py`, `_strava.py`, `_loader.py`, `_parsing.py` |
| 3 | One cleaning definition: pipeline of steps and a strategy ABC | 3.5 | `processing/_pipeline.py`, `_cleaning.py`, `_races.py` |
| 4 | Leakage-safe features and fair evaluation | 3 | `features/_dataset.py`, `evaluation/_splits.py`, `_runner.py`, `models/_baselines.py` |
| 5 | Quality gates that explain themselves | 2.5 | `pyproject.toml`, `tests/test_docstrings.py`, `cli.py` |
| 6 | Live run | 1.5 | terminal |
| 7 | Takeaways | 1 | none |
| | Buffer | 0.5 | |

If you are running late, use the cut list at the end.


## 0. Opening (1.5 min)

**Show:** the project structure block in `README.md`.

**Say:**
- "Race Predictor takes Strava and Garmin exports and predicts running race times. It is a
  thesis project we turned into a package."
- "It is small (about 3,000 lines of source, most of it docstrings), but it has the problems of a bigger one: messy
  inputs, two formats, steps that must be the same in training and prediction, and models that
  must be compared fairly."
- Point at the layout: `sources` (ingest) → `processing` (clean) → `features` (one example per
  race) → `models` and `evaluation`. "Each folder is one stage and each has a small public
  interface."
- Show one `__init__.py` with `__all__`: "Modules are underscore-private (`_base.py`). The
  package decides what is public. Other teams can refactor inside without breaking callers."

**Transition:** "Let me start where most data code goes wrong: passing dicts around."

---

## 1. Objects instead of dicts (3 min)

**Show, in this order:**
1. `domain/_models.py`: `Activity`.
2. `features/_dataset.py`: `TrainingSummary.from_window`, then `RaceFeatures.as_row`.

**Say:**
- `@dataclass(frozen=True)` on `Activity`: "A fixed set of typed fields, generated `__init__`,
  and frozen so it cannot change after creation."
- `__post_init__`: "Validation lives in the object. An `Activity` with a negative distance
  cannot exist, so no later code re-checks. The cleaning pipeline handles *plausibility*, the
  object only guarantees *structure*. That split is why the parser can still represent dirty
  rows."
- `StrEnum` (`ActivityType`): "Members are also strings, so the DataFrame holds plain strings
  and we can still compare with the enum."
- `@property` (`distance_km`, `pace_s_per_km`): derived values, nothing to keep in sync.
- Then the refactor story: "We had a function returning a nine-key dict of floats. Typos in
  keys fail silently, there is no autocomplete and mypy cannot help. We replaced it with
  `TrainingSummary`, `CrossTraining`, `RiegelReference` and `RaceFeatures`."
- `@classmethod from_window`: "Named constructor: how the object is built is in the name."
- `as_row` with `**asdict(...)`: "The objects are flattened into a dict in exactly one place,
  because pandas needs rows. Everywhere else we work with typed objects."
- Same idea in evaluation: `Metrics` and `ModelResult` instead of dicts (`asdict` turns a list
  of them into the results table).

**Takeaway line:** "Use a dict at the boundary (pandas, JSON), objects inside."

**If asked:** "Why not `NamedTuple`/pydantic?" Dataclasses are in the standard library, give
validation hooks, and `asdict` covers the DataFrame conversion. Pydantic is the right choice
when parsing untrusted JSON.

---

## 2. Extensible ingestion (3.5 min)

**Show, in this order:** `sources/_base.py` (`ActivitySource.read`), `_strava.py`
(`parse_row`), `sources/_loader.py` (`detect_source`, `iter_athletes`), `_parsing.py`
(`to_float`).

**Say:**
- "Each export format is an adapter: a subclass of `ActivitySource` that implements two
  methods, `read_table` and `parse_row`."
- `@abstractmethod`: "If someone forgets one, instantiation fails immediately, not halfway
  through a file."
- **Template Method** (the highlight): "`read()` is written once. It checks the format, loops
  over rows, skips the bad ones and logs a single warning with the count. Adapters only
  describe their own format."
- `yield`: "`read` is a generator. Rows stream out one by one, so a 5,000-row export is never
  fully in memory, and `iter_athletes` is lazy for the same reason."
- `@classmethod can_read`: "We sniff the CSV header. It is a classmethod because we ask the
  class before we create any object."
- `frozenset` for `required_columns` and `.issubset(header)`: "All required columns present"
  expressed as set maths.
- `_parsing.py`: "Messy cells (`"6,44"`, `"--"`, empty) become `None`. One convention for a
  missing value, which is what the domain model uses."
- Real-world quirks stay inside the adapter: Garmin Polish headers, pace in min/km for runs but
  km/h for rides, the duplicated heart-rate column in Strava (show the 2-line comment in
  `_strava.read_table`).

**Story worth telling (30 s):** "We first had a registry decorator, `@register_source`, so a
new format needed no other change. In review we asked whether it paid for itself with two
formats. It needed a nested decorator, a `TypeVar` and an import with side effects. We replaced
it with one tuple, `_SOURCES = (StravaCsvSource, GarminCsvSource)`. Adding a format is now two
visible steps instead of magic. Add the decorator back when you have ten formats."

**Takeaway line:** "Put the algorithm in the base class, put the quirks in the subclass, and
make the base class generate lazily."

---

## 3. One cleaning definition for training and prediction (3.5 min)

**Show, in this order:** `processing/_pipeline.py` (`Step`, `Pipeline`, one or two steps),
`processing/_cleaning.py` (`cleaning_pipeline`), `processing/_races.py` (`RaceDetector`,
`HeuristicRaceDetector.reasons`, `DetectRaces`).

**Say:**
- "A classic ML bug: training and the app clean data with different code. A fix goes into one
  and the model sees differently prepared data in production."
- `Step` (ABC with `apply(df) -> df`) and `Pipeline`: "Every step takes a table and returns a
  new one, never modifying its input. That makes each step testable alone, reorderable and
  switchable."
- `Pipeline` has `__len__` and `__iter__` (so `for step in pipeline` works) and a readable
  `__repr__` (print a pipeline to see what it does).
- `cleaning_pipeline(purpose)`: "One function builds both chains. They share the first steps
  and differ only where they must: prediction keeps rows without heart rate and imputes gaps
  *after* race detection, so imputed values can never make a run look like a race."
- `Literal["train", "predict"]` and the bare `*` in the signature: "mypy rejects a typo like
  `"serve"`; we also check at runtime, because users are not type-checked. The `*` makes
  `include_triathlon_races` keyword-only: a bare `True` in a call is unreadable."
- **ABC** (`RaceDetector`): "Race detection is a strategy. Each detector subclasses the base
  class and implements `detect(df) -> Series`. We have the thesis heuristic and a simple
  standard-distance detector."
- One interface mechanism for the whole project: "`Step`, `ActivitySource`, `RaceModel` and
  `RaceDetector` are all ABCs. A `Protocol` would work for detectors, but one mechanism is
  simpler to explain."
- Show `reasons()`: "It returns every scoring condition per run, so you can audit why a run
  was labelled a race. Explainable heuristics beat black boxes when labels are guessed."

**Takeaway line:** "Make the shared definition a function that takes a `purpose`, not two
copies of a notebook cell."

---

## 4. Leakage-safe features and fair evaluation (3 min)

**Show, in this order:** `features/_dataset.py` (`_example`, the `<` on the date),
`evaluation/_splits.py` (both splits), `evaluation/_runner.py` (`compare_models`),
`models/_baselines.py` (`TabularModel`, `riegel_time`).

**Say:**
- **Leakage:** "A race's features use only activities strictly before the race date: `<`, not
  `<=`. One character that decides whether the model is honest. It is covered by a test that
  adds an activity after the race and asserts it does not appear."
- **Two protocols for two questions:** `chronological_split` ("can I predict *this athlete's*
  next race?") and `grouped_splits` ("does it work for someone it has never seen?"), the latter
  via `GroupKFold` so one athlete is never in train and test.
- `RaceModel` (ABC) with `fit(...) -> Self`: "Formula, ridge, gradient boosting and later an
  LSTM all look the same to the evaluation. `Self` keeps `.fit(ds).predict(ds)` typed as the
  concrete model."
- `ModelFactory = Callable[[], RaceModel]`: "We pass *factories*, not models. A fresh unfitted
  model per fold, so nothing leaks between folds. A class is already a zero-argument callable,
  so `DEFAULT_FACTORIES` is just a name-to-class dict."
- `TabularModel` template method: "Subclasses only say which scikit-learn estimator to build."
  `HistGradientBoostingRegressor(random_state=0, **self._params)`: "`**` passes the dict as
  keyword arguments; a fixed seed keeps results reproducible."
- **The generator and timer trick** in `compare_models` (the one line that makes people
  stop): "Fitting happens lazily inside `next(predictions, None)`, so we put the `with
  timer(...)` around `next()` and measure the real fit time per protocol."
- `riegel_time` with a constrained `TypeVar`: "Works on a float, an array or a Series and
  returns the same kind." Mention only if time allows.

**Be honest about the result (10 s):** "On 160 races from 13 athletes, Riegel's 1981 formula is
hard to beat. That is the point of a baseline: the neural networks in the next milestone must
beat it."

**Takeaway line:** "Make leakage a one-line, tested rule, and compare models only through one
interface."

---

## 5. Quality gates that explain themselves (2.5 min)

**Show, in this order:** `pyproject.toml` (ruff `select`, `[tool.mypy]`), `tests/test_docstrings.py`,
`cli.py`.

**Say:**
- "`ruff` for lint and format, `mypy --strict` over `src` *and* `tests`, `pytest` with
  coverage, one `just check` command, the same four checks in GitHub Actions on Python 3.11 and
  3.12."
- Ruff `D` rules with Google-style docstrings: "But ruff only checks *public* functions. So we
  wrote a test, `test_docstrings.py`, that uses the `ast` module to parse every source file and
  fail if any function lacks `Args:` or `Returns:`. About 50 lines, and a rule that cannot
  silently erode."
- `py.typed` + `__all__`: "The package is typed, so other teams get mypy checks on their calls
  to our code."
- `cli.py`: "`argparse` with subcommands, installed as the `race-predictor` command through
  `[project.scripts]`, also `python -m race_predictor` through `__main__.py`. New actions
  (prediction, the app) become subcommands, not loose scripts."
- A word on test style (30 s, pick one): `@pytest.fixture` that returns a *function*
  (`days(n)`, `frame(rows)`, `make_activity(**overrides)`) keeps tests short;
  `@pytest.mark.parametrize` for the negative-value checks; `pytest.approx` for floats.

**Takeaway line:** "Automate the conventions you would otherwise argue about in review."

---

## 6. Live run (1.5 min)

**Run:**
```bash
python3.11 -m race_predictor baselines --models Riegel Ridge --report .artifacts/demo.md
```
**Say while it runs:** "It loads both export folders, cleans, detects races, builds the dataset
and evaluates under both protocols." Point at the table: errors per model and protocol, plus
`fit_s`. Mention that it also wrote `params.json` and `results.csv` to `.artifacts/baselines/`
so runs are reproducible, and that the report keeps its hand-written conclusions when
regenerated (`write_report`).

Fallback if the run fails: open `docs/RESULTS_BASELINES.md` and read the table from there.

---

## 7. Takeaways (1 min)

Say these five, slowly:
1. **Objects inside, dicts at the boundary** (frozen dataclasses with validation).
2. **Base class owns the algorithm, subclasses own the quirks** (ABC and template method,
   lazy generators).
3. **One definition of preparation** for training and prediction (pipeline of small steps,
   `purpose` parameter).
4. **One interface mechanism:** ABCs for every swappable part, factories when you need a fresh
   object.
5. **Make the rules executable:** strict mypy, ruff, a docstring test, a leakage test.

Close with: "We also *removed* things: a registry decorator, `from __future__` imports,
`slots`, `Final`. Advanced Python is knowing which feature pays for itself."

---

## Likely questions and short answers

| Question | Answer |
|---|---|
| Why a frozen dataclass if the DataFrame inside can change? | Frozen blocks reassigning fields. It documents intent and prevents accidental rebinding. It does not deep-freeze. |
| Why not pydantic? | Standard library is enough here. Our data is trusted after parsing, and validation is two checks. |
| Why ABCs and not Protocols? | One mechanism for every interface (`read`, `fit`, `detect`); a missing method fails at instantiation. A Protocol would also work for `detect`. |
| Why strict mypy on tests too? | Fixtures and helpers are code. Typed tests catch API drift when signatures change. |
| Is the pipeline over-engineered for pandas? | Each step is ~10 lines and tested alone. The payoff is the `train`/`predict` split, which would otherwise be two copies. |
| Why generators in ingestion? | Memory and early exit. `read` and `iter_athletes` yield one item at a time. |
| Why remove the registry? | With two formats it cost a nested decorator, a `TypeVar` and a hidden import for no benefit. Bring it back at ~10 formats. |
| Does the model beat the baseline? | Not yet. Riegel is hard to beat on 160 races. Sequence models are the next milestone. |

---

## Cut list (if you are behind schedule)

Cut in this order, each saves about the time shown:
1. The `riegel_time` / `TypeVar` remark in section 4 (20 s).
2. Test-style paragraph in section 5 (30 s).
3. `reasons()` in section 3 (30 s).
4. The registry story in section 2, keep just the last sentence (30 s).
5. Shorten the live run to showing the saved `RESULTS_BASELINES.md` (1 min).

## Add list (if you finish early)

- Show `Athlete.runs()` (`sorted(..., key=lambda ...)`) as a tiny example of a generator
  expression feeding `sorted`.
- Show `utils.timer` as a `@contextmanager`: setup, `yield`, `finally`.
- Show `tests/features/test_dataset.py` to prove the no-leakage rule.

## Backup material

`docs/PYTHON_CHEATSHEET.md` explains every construct used above (cls vs self, `Self`,
`TypeVar`, `**`, `yield`, ...) in one-sentence answers. Skim it the evening before.
