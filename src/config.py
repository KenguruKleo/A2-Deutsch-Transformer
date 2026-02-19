import yaml
from pathlib import Path

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
