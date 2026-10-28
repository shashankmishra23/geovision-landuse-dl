"""Config loading, CLI overrides, seeding and device selection."""

import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=None, overrides=None):
    """Load YAML config. `overrides` is a list like ["train.epochs=5", "model.name=resnet50"]."""
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)  # "5" -> 5, "true" -> True, "null" -> None
    return cfg


def resolve(path):
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda") + 1  # can be "available" yet unusable (e.g. driver too old)
            return torch.device("cuda")
        except RuntimeError:
            pass
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
