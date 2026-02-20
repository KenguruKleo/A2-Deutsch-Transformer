#!/usr/bin/env bash
# Pre-commit hook: regenerate hf_export/ from model_final.pth and stage it.
# Run from repo root; use venv Python so IDE git has same env as terminal.
set -e
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
if [ ! -f model_final.pth ]; then
  echo "⏭ model_final.pth not found — skipping HF export (run: python src/train.py)"
  exit 0
fi
if [ -x ".venv/bin/python" ]; then
  .venv/bin/python src/export_hf.py
else
  python src/export_hf.py
fi
git add hf_export/
