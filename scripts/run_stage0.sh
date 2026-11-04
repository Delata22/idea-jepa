#!/usr/bin/env bash
# Stage 0: environment, data generation, unit tests.
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync
uv run python -m ideajepa.data.synthetic
uv run python -m ideajepa.data.synthetic_entropy
uv run pytest -q
