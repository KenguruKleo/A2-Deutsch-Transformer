import yaml
import torch
from pathlib import Path


def _xpu_available() -> bool:
    """Intel XPU may be missing in some PyTorch builds."""
    xpu = getattr(torch, "xpu", None)
    return xpu is not None and xpu.is_available()


def get_device(device_preference: str | None = None) -> str:
    """
    Resolve the device to use. If preference is "auto" or None, picks the best
    available: cuda → xpu → mps → cpu. Otherwise uses the given preference with
    fallback to cpu when the chosen device is not available.

    Supported devices: "cuda" (NVIDIA), "xpu" (Intel), "mps" (Apple Silicon), "cpu".
    """
    if device_preference and device_preference != "auto":
        # Explicit choice: validate availability
        if device_preference == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device_preference == "xpu" and _xpu_available():
            return "xpu"
        if device_preference == "mps" and torch.backends.mps.is_available():
            return "mps"
        if device_preference == "cpu":
            return "cpu"
        # Requested device not available → fallback to cpu
        return "cpu"
    # Auto: pick best available (cuda → xpu → mps → cpu)
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Config:
    """Namespace to store and access configuration attributes."""
    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(path: str = "config.yaml") -> Config:
    """Loads YAML config and returns a Config object."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    return Config(data)
