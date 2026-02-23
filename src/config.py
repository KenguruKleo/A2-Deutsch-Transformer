import yaml
import torch
from dataclasses import dataclass
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


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    weight_tying: bool = True


@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    early_stopping_patience: int = 0
    decision_token_weight: float = 1.0
    warmup_steps: int = 0
    device: str = "auto"


@dataclass
class DataConfig:
    train_path: str
    val_path: str


@dataclass
class GenerationConfig:
    temperature: float
    top_k: int


@dataclass
class Config:
    """Typed config matching config.yaml structure."""
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    generation: GenerationConfig

# Project root is two levels up from this file (src/config.py → src/ → project root)
_PROJECT_ROOT = Path(__file__).parent.parent


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return _PROJECT_ROOT


def load_config(path: str | Path | None = None) -> Config:
    """Loads YAML config and returns a typed Config object.

    If path is not provided, looks for config.yaml in the project root.
    """
    if path is None:
        config_path = _PROJECT_ROOT / "config.yaml"
    else:
        config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return Config(
        model=ModelConfig(**data["model"]),
        training=TrainingConfig(**data["training"]),
        data=DataConfig(**data["data"]),
        generation=GenerationConfig(**data["generation"]),
    )
