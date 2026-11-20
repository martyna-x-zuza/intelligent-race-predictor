# Python cheatsheet for this repository

Every "advanced" construct in the code, what it is, and why it is used here. Each entry has a
one-sentence answer you can say out loud, then the details. Examples are taken from the code.

Sections: 1. Classes and decorators · 2. Dunder methods · 3. Typing · 4. Collections and
iteration · 5. numpy · 6. Functions and arguments · 7. Standard library · 8. Tests · 9. Things
you did not list · 10. Honest opinion: what could still be simpler

---

## 1. Classes and decorators

### `cls` versus `self`
*One sentence: `self` is "this object", `cls` is "this class".*

| | `self` | `cls` |
|---|---|---|
| Method kind | normal (instance) method | `@classmethod` |
| Needs an object first? | yes: `athlete.runs()` | no: `RaceCategory.from_distance(5000)` |
| Typical use | read or change this object's data | build an object, or work with class-level data |

`detect_source` calls `source.can_read(path)` on the **class** (`StravaCsvSource`) before any
object exists. That works because `can_read` is a classmethod.

### `@classmethod`
*Method that receives the class instead of an object. Used as an alternative constructor.*
- `RaceCategory.from_distance(distance_m)`, `Race.from_activity(activity)`,
  `TrainingSummary.from_window(window, race_date, days_window)`,
  `RiegelReference.from_window(window)`, `ActivitySource.can_read(path)`.
- Why: the name says how the object is built ("from a window"), and the logic stays next to the
  class. `return cls(...)` creates an instance of the class it was called on.

### `@property`
*Method you read like an attribute: `activity.distance_km`, no parentheses.*
- Used for values derived from other fields: `Activity.distance_km`, `pace_s_per_km`,
  `Race.time_s`, `Athlete.hr_max`, `RaceDataset.race_time_s`.
- Why: nothing to store or keep in sync, and it is always up to date.
- Convention: a property for a cheap derived value, a normal method (`Athlete.runs()`) when it
  does real work or builds a new list.

### `@dataclass(frozen=True)`
*Class whose boilerplate (`__init__`, `__repr__`, `__eq__`) is generated from the annotated fields.*
```python
@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
```
- Without it you would write `__init__` with `self.mae = mae` ... for every field.
- `frozen=True`: fields cannot be reassigned after creation (`m.mae = 1` raises
  `FrozenInstanceError`). The object is also hashable.
- Why frozen: `Activity` is a record of something that already happened, so validated data
  stays valid. Results (`Metrics`, `ModelResult`) and feature objects never need changing.
- `Athlete` is **not** frozen because `add_activity` modifies it.
- Frozen only blocks reassigning attributes. A DataFrame inside a frozen object can still be
  modified in place. `RaceDataset` is frozen but holds DataFrames.

### `__post_init__(self)`
*Hook the dataclass calls right after the generated `__init__`. Used for validation.*
- `Activity` rejects an empty `athlete_id` and negative quantities. `RaceDataset` checks that
  all parts have the same number of races.
- Why: invalid objects cannot exist, so later code does not have to re-check.

### `@abstractmethod` and `from abc import ABC`
*`ABC` marks a base class that is not meant to be used directly. `@abstractmethod` marks methods
subclasses must implement.*
- `Step.apply`, `ActivitySource.read_table` and `parse_row`, `RaceModel.fit` and `predict`.
- Creating `Step()` or a subclass that forgot `apply` raises `TypeError`.
- Why: defines the contract ("every model has `fit` and `predict`"). It also enables the
  Template Method pattern: `ActivitySource.read()` is written once and calls the abstract
  `read_table` and `parse_row`.

### `ClassVar` and `name = "Riegel"`
*A class variable is one value shared by the class (and all its instances), not stored per object.*
```python
class RaceModel(ABC):
    name: ClassVar[str]  # declaration only: "every subclass must set it"


class RiegelBaseline(RaceModel):
    name = "Riegel"  # the value
```
- `ClassVar` tells mypy "this is a class-level attribute, not an instance field". It is used in
  the base classes only.
- `name` is used in the result tables and error messages. The same pattern holds for
  `ActivitySource.name` and `required_columns`.

### `Self` (`from typing import ClassVar, Self`)
*"The same type as the object this method was called on."*
- `RaceModel.fit(self, ds) -> Self`: `RidgeBaseline().fit(ds)` is typed as `RidgeBaseline`, not
  just `RaceModel`, so `.fit(ds).predict(ds)` chains type-correctly.
- It also avoids writing `-> "RaceModel"` inside `RaceModel` itself.
- The alternative constructors (`from_window`, `from_distance`, `from_activity`, ...) return
  `Self` too: they end in `cls(...)`, so the result is an instance of the class they were
  called on, and no quoted forward reference is needed.
- `RaceDataset.subset` is an instance method, so there is no `cls`: it returns
  `dataclasses.replace(self, ...)`, a modified copy of the frozen instance of the same class
  (it re-runs `__post_init__`), and is typed `-> Self`.


### `name`, `_private`, `__dunder__`
- `_name`: private by convention (modules `_base.py`, functions `_total`, attributes
  `self._params`). Not enforced; it means "internal, don't use from outside".
- `__name__`: dunder = double underscore, a special name Python itself uses.

---

## 2. Dunder (special) methods

Python calls these for you. You never write `obj.__len__()`.

| Method | Triggered by | Where / why |
|---|---|---|
| `__init__` | `Foo(...)` | generated by `@dataclass`, written by hand elsewhere |
| `__post_init__` | right after `__init__` (dataclass only) | validation |
| `__len__(self)` | `len(obj)` | `Pipeline` (number of steps), `RaceDataset` (number of races) |
| `__iter__(self) -> Iterator[Step]` | `for step in pipeline` | `Pipeline` returns `iter(self._steps)`, so it can be looped over directly |
| `__repr__(self)` | `repr(obj)`, printing in the REPL, test failure output | `Step`, `Pipeline`, `RaceModel`, `DetectRaces` show readable text like `RidgeBaseline()`. Tests assert on it. |

---

## 3. Typing

Type hints do nothing at runtime. **mypy** reads them (the project uses `strict = true`).

### `Callable[[], RaceModel]` and `ModelFactory = ...`
```python
ModelFactory = Callable[[], RaceModel]
```
- `Callable[[args], result]` = a function type. `[]` = takes no arguments, returns a
  `RaceModel`.
- `ModelFactory = ...` is a **type alias**: just a shorter name.
- A **class is callable**: `RiegelBaseline()` creates an object, so the class itself fits
  `ModelFactory`.
- Why a factory and not an instance: every split needs a fresh, unfitted model (`make()` is
  called per fold). A reused instance would carry what it learned in the previous split.
- `Callable[..., Activity]` (in `tests/conftest.py`): `...` means "any arguments".

### `DEFAULT_FACTORIES: Mapping[str, ModelFactory] = {...}`
```python
DEFAULT_FACTORIES = {
    "Riegel": RiegelBaseline,
    "Ridge": RidgeBaseline,
    "HistGB": GradientBoostingBaseline,
}
```
- Name to class (not an instance: no parentheses!). One place that lists the available models.
- The CLI uses its keys for `--models` choices, and `run_baselines` calls the values to build
  models.
- `Mapping` = read-only dict type ("I only read it, nobody modifies it").

### `models: tuple[str, ...]`
- A tuple of strings of **any length**. `tuple[str]` would mean exactly one item, and `...`
  means "variable length".
- Why a tuple, not a list: the dataclass is `frozen`, so its content should be immutable too.

### `Purpose = Literal["train", "predict"]`
- `Literal` restricts a value to exact constants. mypy rejects `cleaning_pipeline("serve")`.
- The function also checks at runtime (`raise ValueError`), because at runtime Python ignores
  hints (a CLI user could pass anything). The test calls it with `# type: ignore[arg-type]` to
  prove this.

### `_Number = TypeVar("_Number", float, npt.NDArray[np.float64], "pd.Series[float]")`
*A placeholder type: "whatever kind goes in is the kind that comes out".*
- `riegel_time(...)` works on a float, a numpy array or a pandas Series. A `TypeVar` with
  listed constraints says: all arguments are the same one of these kinds, and the result is
  that kind.
- `"pd.Series[float]"` is quoted because `pd.Series[float]` cannot be evaluated at runtime
  (only mypy understands it).
- The `_` prefix: private helper type.

### `npt.NDArray[np.float64]`, `Index = npt.NDArray[np.intp]`
- `import numpy.typing as npt`.
- `NDArray[np.float64]` = numpy array of 64-bit floats (`FloatArray` in the code).
- `evaluate` takes `FloatArray` arguments, and `subset` takes `Index | npt.NDArray[np.bool_]`
  (integer positions or a boolean mask). Tests pass `np.array(...)` instead of plain lists, so
  the annotations say exactly what is accepted.
- `Index = npt.NDArray[np.intp]` is an alias: an array of integer positions (see `np.intp`
  below). It is defined in `features` (next to `FloatArray`) because `subset` uses it and
  `evaluation` depends on `features`, not the other way round.

### `X | None`, `X | Y`
- Union types: `float | None` = a float or `None` (a missing measurement).
- Also works at runtime (3.10+): `isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)`.

### `Any`
- "Turn off type checking here". Used where libraries have no type hints: scikit-learn
  estimators (`self._estimator: Any`), mixed dicts in tests.

### `Hashable`, `Mapping`, `Row = Mapping[Hashable, Any]`
- `Hashable`: can be a dict key (has `__hash__`: str, int, tuple... not list).
- pandas types `DataFrame.to_dict(orient="records")` as `dict[Hashable, Any]`, so the row alias
  has to accept `Hashable` keys. `Row` is "one export row as a column-to-value mapping".

### `Sequence`, `Iterable`, `Iterator`, `Generator`
From general to specific:

| Type | You can | Example |
|---|---|---|
| `Iterable[T]` | loop over it | `Pipeline(steps: Iterable[Step])` accepts a list, tuple, generator |
| `Iterator[T]` | loop **once**, `next()` | return type of generator functions that `yield` |
| `Sequence[T]` | loop, index `x[0]`, `len()` | `main(argv: Sequence[str] | None)` |
| `Generator[Y, S, R]` | an Iterator that also has send/return types | `timer()` is `Generator[Timing, None, None]` |

Rule of thumb: ask for the most general type your code needs. Taking `Sequence` means "I need
indexing", taking `Iterable` means "I only loop".

### `argparse.Namespace`
- What `parser.parse_args()` returns: an object whose attributes are your arguments
  (`args.data`, `args.n_splits`; the CLI flag `--n-splits` becomes `n_splits`).
- Used as the type of `_baselines(args)` in `cli.py`.

### `type: ignore[arg-type]`
- Silences one specific mypy error. Used once, in a test that deliberately passes a wrong value.

---

## 4. Collections and iteration

### `frozenset`
*An immutable set.*
- `required_columns = frozenset({...})`, `_REQUIRED_COLUMNS`, `_MISSING`.
- Sets give fast membership (`x in s`) and set maths: `required_columns.issubset(header)`
  ("all required columns are present"), `_REQUIRED_COLUMNS - set(df.columns)` ("which are
  missing").
- Frozen so a constant cannot be modified by accident.

### `yield` (generators)
```python
def training_windows(df, ...) -> Iterator[tuple[pd.Series, pd.DataFrame]]:
    ...
    yield race, window
```
- A function with `yield` returns a **generator**: values are produced one at a time, when
  asked for. It pauses at each `yield` and continues from there.
- Why: lazy, so big exports are not all held in memory (`ActivitySource.read`,
  `iter_athletes`), and the caller can stop early.
- `yield race, window` yields one tuple of two values.
- `next(gen, None)`: take the next value or `None` if finished (no `StopIteration`). In
  `compare_models`, because fitting happens lazily inside `next()`, the timer wraps `next()` and
  measures the real fit time.

### `COLUMNS = ("source", *(f.name for f in fields(Activity)))`
Read from the inside out:
1. `fields(Activity)`: the dataclass's fields (`from dataclasses import fields`).
2. `(f.name for f in ...)`: a generator expression giving each field's name.
3. `*( ... )`: unpacks it into the tuple, after `"source"`.
- Result: `("source", "athlete_id", "date", ...)`. The table columns follow the dataclass
  automatically, so a new field cannot be forgotten.

### `**` (dict unpacking) and `asdict`
```python
return {
    "years_of_experience": self.years_of_experience,
    ...
    **asdict(self.cross_training),
    **asdict(self.training),
    **asdict(self.riegel),
}
```
- `asdict(obj)` turns a dataclass instance into a dict (`{"swim_t_total": 0.0, ...}`).
- `**d` inside a dict literal copies all of its entries in. If keys repeat, the later one wins.
- Why: the three small objects become **one flat row** with the old column names. `as_row()` is
  the only place where objects become a dict (needed by `pd.DataFrame`).
- `{**defaults, **overrides}` (tests): merge two dicts, overrides win.

### `**self._params` and `random_state=0`
```python
HistGradientBoostingRegressor(random_state=0, **self._params)
```
- In a **call**, `**dict` passes each entry as a keyword argument:
  `learning_rate=0.05, max_iter=200, ...`.
- `random_state=0` fixes the seed so results are reproducible.

### Comprehensions and generator expressions
- `[x for x in items if cond]` builds a list, `{...}` a set or dict, `(x for x in items)` a lazy
  generator. Used in many places (`sorted(a for a in ... if ...)`).

### `lambda`
- A one-line anonymous function: `key=lambda a: a.date` (sort by date). In the test fixtures it
  is returned as a tiny helper: `days(n)`.

### `is` versus `==`
- `a.type is ActivityType.RUN`: identity check, used for enum members (there is exactly one
  `RUN`). `==` compares values.

---

## 5. numpy

### `np.flatnonzero(mask)`
- Returns the **indices** where a boolean array is `True`: `flatnonzero([F, T, T, F]) -> [1, 2]`.
- In `chronological_split`: `np.flatnonzero(~test), np.flatnonzero(test)` gives the train
  indices and the test indices (`~` inverts a boolean array).
- In `RaceDataset.subset`: turns a boolean mask into positions.

### `np.intp`
- The integer type numpy uses for **indexing** (64-bit on a 64-bit machine). Splits are arrays
  of positions, so `Index = npt.NDArray[np.intp]`.
- `np.asarray(train, dtype=np.intp)` in `grouped_splits` makes the type of the indices
  explicit, so it matches the `Index` alias that mypy checks.

### Others you will meet
- `np.asarray(x, dtype=float)`: convert to an array without copying if it already is one.
- `np.where(cond, a, b)`: element-wise "if cond then a else b" (the Riegel fallback).
- `np.nan`: "not a number", pandas' missing value. `np.isnan(...)` tests for it.
- `np.full(n, value)`: array of `n` copies.

---

## 6. Functions and arguments

### The bare `*` in a signature
```python
def cleaning_pipeline(
    purpose: Purpose = "train",
    detector: RaceDetector | None = None,
    *,
    include_triathlon_races: bool = False,
)
```
- Everything after `*` is **keyword-only**: `cleaning_pipeline("train", None, include_triathlon_races=True)`.
- Why: a bare `True` as a third argument would be unreadable, and keywords make call sites
  self-explanatory.

### Default arguments
- `purpose: Purpose = "train"`. Defaults must be immutable; mutable ones (lists) are shared
  between calls. The code uses `None` and builds the object inside (`detector or
  HeuristicRaceDetector()`).

### `**overrides: Any`
- Collects extra keyword arguments into a dict (test factory `make_activity(distance_m=5000)`).

### `with ... as`
- Context manager: setup and cleanup around a block. `with timer("fit") as elapsed:` starts a
  stopwatch and stops it even if the block raises.
- `timer` is a generator function with `@contextmanager`: code before `yield` is setup, after
  it is cleanup. `yield` hands over the `Timing` object.

---

## 7. Standard library

### `_logger = logging.getLogger(__name__)`
- `__name__` is the module's dotted name (for example `race_predictor.sources._loader`), so log
  lines say where they came from.
- Underscore: private to the module. Used for warnings like "skipped 2 unreadable row(s)".
- The CLI decides what is shown: `-v` shows warnings.

### `ABC` + `@abstractmethod` (`RaceDetector`)
- Same mechanism as `RaceModel` and `ActivitySource`: the base class declares
  `detect(df) -> Series`, and `HeuristicRaceDetector` and `StandardDistanceDetector` subclass
  it. A subclass that forgets `detect` cannot be instantiated.
- `DetectRaces(detector: RaceDetector)` accepts any subclass (Strategy pattern).
- A `Protocol` would also work (no inheritance needed, static duck typing); an ABC was chosen so
  the project has one interface mechanism.

### `StrEnum` (`ActivityType`, `RaceCategory`)
- An enum whose members are also strings: `ActivityType.RUN == "Run"` is `True`. That is why the
  DataFrame can hold plain strings in the `type` column and still be compared with the enum.

### `ast` (in `tests/test_docstrings.py`)
- Parses Python source into a tree without running it. The test walks every function in `src/`
  and checks that it has a docstring with `Args:` and `Returns:`.
- `FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef`: a type alias for "a def or an
  async def".

### `__all__` and `_module.py` names
- `__all__` in each `__init__.py` lists the public API. Modules like `_base.py` are internal;
  users import from the package (`from race_predictor.sources import detect_source`).

---

## 8. Tests (pytest)

### `@pytest.fixture`
- A function whose result is **injected** into tests by parameter name:
  ```python
  @pytest.fixture
  def days() -> Days: ...


  def test_x(days: Days) -> None: ...  # pytest passes the fixture's value
  ```
- Fixtures in `tests/conftest.py` are available to all tests. They remove repeated setup.

### `Days`, `Frame`, `ActivityFactory` (and `from tests.conftest import Days, Frame`)
```python
Days = Callable[[float], pd.Timestamp]
```
- Our fixtures return **small functions** (a lambda): `days(3)` gives the date 3 days after
  2024-01-01, `frame(rows)` builds a table from row overrides, `make_activity(distance_m=...)`
  builds an `Activity`.
- The aliases only exist so mypy strict can type the test parameters. They live in `conftest.py`
  and tests import them for annotations.

### `@pytest.mark.parametrize`
```python
@pytest.mark.parametrize(
    "field",
    ["distance_m", "elapsed_s", "moving_s", ...],
)
def test_negative_quantities_are_rejected(make_activity, field: str) -> None: ...
```
- Runs the same test once per value. Each value is reported as its own test, so you see which
  one fails. With two names (`("y_true", "y_pred")`), every case is a tuple.

### `pytest.approx(10.5)`
- Compares floats with a tolerance: `assert x == pytest.approx(10.5)`. Needed because
  `0.1 + 0.2 != 0.3` in floating point. `abs=0.01` sets an absolute tolerance.

### `pytest.raises(Error, match=...)`
- Asserts that the block raises that error and that the message matches a regex.

### `tmp_path`, `caplog`, `capsys`
- Built-in fixtures: a temporary directory, captured log records, captured printed output.

---

## 9. Things you did not list (a teacher might ask)

| Term | Where | One-sentence answer |
|---|---|---|
| `@contextmanager` | `utils.timer` | turns a generator with one `yield` into a `with` block |
| `ABC` | `RaceDetector` | interface enforced by inheritance |
| `StrEnum` | `ActivityType`, `RaceCategory` | enum members that are also strings |
| Template Method | `ActivitySource.read`, `TabularModel` | base class fixes the algorithm, subclasses fill in steps |
| Strategy | `RaceDetector`, `DetectRaces(detector)` | swap the algorithm by passing a different object |
| Pipeline | `Pipeline`, `Step` | chain of small steps, each testable alone |
| Adapter | `StravaCsvSource`, `GarminCsvSource` | different formats converted to one `Activity` type |
| `try/except ... else/finally` | `timer`, parsing helpers | `finally` always runs, so timing is recorded even on errors |
| `raise X(...) from None` | error handling | hides an irrelevant earlier exception |
| `sorted(..., key=...)` | `Athlete.runs` | sort by a computed value |
| `DataFrame.groupby` | `_pipeline.py`, `_dataset.py` | split by athlete, apply the same logic to each |
| `# noqa`, `# type: ignore` | few places | silence a linter or mypy rule on one line on purpose |
| `pyproject.toml` `[project.scripts]` | `race-predictor` command | installs `cli:main` as a terminal command |
| `__main__.py` | `python -m race_predictor` | the file that runs when the package is run as a program |

---

## 10. Honest opinion: what could still be simpler

If a teacher asks "is this needed?", these are the ones I would be ready to defend or drop:

1. **`__repr__` on `Step`, `Pipeline`, `RaceModel`, `DetectRaces`.** Convenient for debugging,
   and tests assert on them, but the program itself never needs them. Easy to defend ("readable
   output when debugging"), easy to drop.
2. **`_Number = TypeVar(...)`.** It exists so `riegel_time` accepts floats, arrays and Series.
   In the program only arrays are used. The tests use floats too. If you simplify, the
   function could take arrays only, and the TypeVar would go away.
3. **`Purpose = Literal[...]` plus the runtime check.** The check is the one that protects
   users. The `Literal` additionally helps mypy and editors. Keeping both is slightly
   redundant but cheap.
4. **ABC for detectors.** The base class has no shared code, only the contract, so a
   `Protocol` would also do. An ABC keeps one interface mechanism across the project.
5. **`@dataclass(frozen=True)` on `RaceDataset`.** It protects the attributes but not the
   DataFrames inside. Say "frozen means the fields cannot be reassigned", not "the data is
   immutable".
6. **`Frame` / `Days` as lambdas in fixtures.** They are compact but a little unusual. A plain
   helper function in `conftest.py` would do the same.
7. **`ClassVar` only in base classes.** If asked why subclasses write `name = "Riegel"`
   without it: a plain assignment in a subclass is a class attribute already, and mypy knows it
   from the base declaration.
