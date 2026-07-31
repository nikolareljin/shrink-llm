# Contributing

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected environment:

- Python 3.10, 3.11 or 3.12. CI runs all three, so any of them is a fair match. Note that the
  test suite parses `pyproject.toml`, and `tomllib` is stdlib only from 3.11 — the `dev` extra
  supplies `tomli` below that.
- Large models and datasets stored outside the tracked repository tree unless a placeholder file is required.
- The `scripts/script-helpers` submodule is declared in `.gitmodules` but currently unused: the
  repository contains no shell scripts. Initializing it is optional —
  `git submodule update --init --recursive`.

Optional extras:

```bash
pip install -e ".[dev,tflite]"
pip install -e ".[dev,coreml]"
pip install -e ".[dev,gptq,audio]"
```

## Common commands

These must stay identical to CI's. Linting a narrower tree than CI does means passing locally and
failing the gate.

```bash
pytest -q
ruff check scripts/ compression/ benchmarks/ mobile_deployment/ tests/
black --check scripts/ compression/ benchmarks/ mobile_deployment/ tests/
python scripts/run_pipeline.py --config configs/ocr_pipeline.yaml --dry-run
```

Bootstrap after cloning:

```bash
git clone https://github.com/nikolareljin/shrink-llm.git
cd shrink-llm
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Repository conventions

- Keep large model weights and raw datasets out of git.
- Treat `TODO.txt` as a local planning file; it is intentionally ignored.
- Prefer adding reproducible scripts and small manifests instead of committing generated artifacts.
