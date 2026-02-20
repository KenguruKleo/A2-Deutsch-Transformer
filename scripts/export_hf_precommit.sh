#!/usr/bin/env bash
# Pre-commit hook: regenerate hf_export/ from model_final.pth and stage it.
set -e
if [ ! -f model_final.pth ]; then
  echo "⏭ model_final.pth not found — skipping HF export (run: python src/train.py)"
  exit 0
fi
python src/export_hf.py
git add hf_export/
