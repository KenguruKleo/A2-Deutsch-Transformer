"""Upload hf_export/ to Hugging Face Hub. Uses env TOKEN (HF_TOKEN or HG_TOKEN)."""
import os
from huggingface_hub import HfApi

token = os.environ.get("TOKEN", "").strip() or None
api = HfApi()
api.upload_folder(
    folder_path="hf_export",
    repo_id="kengurukleo/deutsch_a2_transformer",
    repo_type="model",
    token=token,
)
print("Uploaded hf_export/ to kengurukleo/deutsch_a2_transformer")
