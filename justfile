# Entry point of the repository: every development and project task starts here.
# Run `just` to list the recipes. `just` itself is installed with the dev dependencies.
# Tools write their caches and outputs to .artifacts/<tool>/ (see pyproject.toml).

# List the available recipes
default:
    @just --list

# Install the package in editable mode with the development tools
install:
    pip install -e ".[dev]"

# Lint and check formatting (ruff)
lint:
    ruff check .
    ruff format --check .

# Reformat the code and apply automatic lint fixes
format:
    ruff format .
    ruff check --fix .

# Type check src/ and tests/ (mypy, strict)
typecheck:
    mypy

# Run the tests with coverage; extra arguments go to pytest, e.g. `just test -k sources`
test *args:
    pytest {{ args }}

# Everything CI runs: lint, type check and tests
check: lint typecheck test

# Evaluate the baselines: report in docs/, parameters and raw results in .artifacts/baselines/
baselines *args:
    race-predictor baselines {{ args }}

# Compare the GRU and LSTM with the baselines (needs the deep extra): docs/RESULTS_SEQUENCE.md
sequence:
    race-predictor baselines --models Riegel Ridge GRU LSTM \
        --report docs/RESULTS_SEQUENCE.md --artifacts .artifacts/sequence

# Start the Streamlit app (needs the app extra)
app:
    streamlit run src/race_predictor/app.py

# Delete all tool caches and outputs
clean:
    rm -rf .artifacts
