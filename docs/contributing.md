# Contributing

## Local setup

```bash
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Expected environment:

- Python 3.10+ locally. CI runs on Python 3.11, so use 3.11 when you want the closest match.
- A checked-out `scripts/script-helpers` submodule before running shell helpers or CI-like workflows.
- Large models and datasets stored outside the tracked repository tree unless a placeholder file is required.

Optional extras:

```bash
pip install -e ".[dev,tflite]"
pip install -e ".[dev,coreml]"
pip install -e ".[dev,gptq,audio]"
```

## Common commands

```bash
pytest -q
ruff check scripts/ compression/ benchmarks/ tests/
black --check scripts/ compression/ benchmarks/ tests/
python scripts/run_pipeline.py --config configs/ocr_pipeline.yaml --dry-run
```

Bootstrap after cloning:

```bash
git clone https://github.com/nikolareljin/shrink-llm.git
cd shrink-llm
git submodule update --init --recursive
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Repository conventions

- Keep large model weights and raw datasets out of git.
- Treat `TODO.txt` as a local planning file; it is intentionally ignored.
- Prefer adding reproducible scripts and small manifests instead of committing generated artifacts.
