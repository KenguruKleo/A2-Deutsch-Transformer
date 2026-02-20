"""Upload hf_space/ to Hugging Face Space. Uses env TOKEN (HF_TOKEN or HG_TOKEN)."""
import os
from pathlib import Path

from huggingface_hub import HfApi

token = os.environ.get("TOKEN", "").strip() or None
if not token:
    raise SystemExit("Set TOKEN (or HF_TOKEN/HG_TOKEN) to upload the Space.")

repo_root = Path(__file__).resolve().parent.parent
folder_path = repo_root / "hf_space"
if not folder_path.is_dir():
    raise SystemExit("hf_space/ not found.")

api = HfApi(token=token)
api.upload_folder(
    folder_path=str(folder_path),
    repo_id="kengurukleo/deutsch-a2-tutor",
    repo_type="space",
    token=token,
)
print("Uploaded hf_space/ to kengurukleo/deutsch-a2-tutor (Space)")
