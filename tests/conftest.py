"""Shared fixtures: a tiny synthetic dataset and an untrained checkpoint, so tests never need real data."""

import numpy as np
import pytest
import torch
from PIL import Image

from geovision.data import IMAGENET_MEAN, IMAGENET_STD, build_manifest
from geovision.models import build_model

CLASSES = ["Forest", "Industrial", "SeaLake"]


@pytest.fixture(scope="session")
def fake_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("eurosat")
    rng = np.random.default_rng(0)
    base = {"Forest": (40, 90, 40), "Industrial": (140, 140, 140), "SeaLake": (30, 60, 120)}
    for cls in CLASSES:
        (root / cls).mkdir()
        for i in range(20):
            arr = np.clip(rng.normal(base[cls], 15, (64, 64, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / cls / f"{cls}_{i}.jpg")
    (root / "Forest" / "broken.jpg").write_bytes(b"not an image")
    return root


@pytest.fixture(scope="session")
def manifest(fake_dataset, tmp_path_factory):
    path = tmp_path_factory.mktemp("meta") / "manifest.csv"
    df, classes, report = build_manifest(fake_dataset, path, 0.2, 0.2, seed=0)
    return path, df, classes, report


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory):
    model = build_model("baseline_cnn", len(CLASSES), pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    torch.save({"state_dict": model.state_dict(), "model_name": "baseline_cnn", "classes": CLASSES,
                "img_size": 64, "dropout": 0.3, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                "run_id": "test-run", "version": "test", "val_metrics": {"accuracy": 0.5, "macro_f1": 0.5}}, path)
    return path
