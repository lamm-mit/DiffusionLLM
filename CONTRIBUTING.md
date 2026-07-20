# Contributing

Thank you for helping improve DiffusionLLM.

## Development setup

```bash
git clone https://github.com/lamm-mit/DiffusionLLM.git
cd DiffusionLLM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Before opening a pull request

Run the full local checks:

```bash
PYTHONPATH=src pytest
ruff check .
```

Changes to model attention, corruption, or sampling should include a focused
test. New model families must demonstrate that future tokens affect past logits
and that padding does not change logits at real-token positions.

Keep pull requests focused and explain:

- what changed;
- why it is needed;
- how it affects students or researchers; and
- which commands were used to validate it.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0.
