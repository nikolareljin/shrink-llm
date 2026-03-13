# Contributing

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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

## Repository conventions

- Keep large model weights and raw datasets out of git.
- Treat `TODO.txt` as a local planning file; it is intentionally ignored.
- Prefer adding reproducible scripts and small manifests instead of committing generated artifacts.
