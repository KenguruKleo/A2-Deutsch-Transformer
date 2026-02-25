"""Restart HF Space. Uses env TOKEN (HF_TOKEN or HG_TOKEN)."""
import os
from huggingface_hub import HfApi

token = os.environ.get("HF_TOKEN") or os.environ.get("HG_TOKEN") or os.environ.get("TOKEN")
if not token:
    print("No token found — skipping Space restart.")
else:
    api = HfApi(token=token)
    api.restart_space(repo_id="kengurukleo/deutsch-a2-tutor")
    print("✅ Space restart requested.")
