"""Data layer: download, validation, stratified splitting, transforms and loaders.

Raw data  : EuroSAT RGB (Sentinel-2 satellite tiles, 27,000 images, 10 land-use classes)
Pipeline  : download -> integrity check -> stratified train/val/test manifest -> Dataset/DataLoader
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .config import resolve

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


# --------------------------------------------------------------------------- download

def download_eurosat(root):
    """Download EuroSAT RGB via torchvision and return the folder with one sub-folder per class."""
    root = resolve(root)
    root.mkdir(parents=True, exist_ok=True)
    datasets.EuroSAT(root=str(root), download=True)
    image_dir = root / "eurosat" / "2750"
    if not image_dir.exists():
        raise FileNotFoundError(f"Expected EuroSAT images at {image_dir}")
    return image_dir


# --------------------------------------------------------------------------- validation

def image_stats(img):
    """Cheap per-image statistics. Used both for EDA and for production drift monitoring."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
    gray = arr.mean(axis=2)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "r_mean": float(r),
        "g_mean": float(g),
        "b_mean": float(b),
    }


def scan_images(image_dir):
    """Walk class folders, drop corrupt/unreadable files and collect stats for the data-quality report."""
    image_dir = Path(image_dir)
    classes = sorted(d.name for d in image_dir.iterdir() if d.is_dir())
    records, corrupt = [], []
    for label, cls in enumerate(classes):
        for path in sorted((image_dir / cls).iterdir()):
            if path.suffix.lower() not in IMG_EXTS:
                continue
            try:
                with Image.open(path) as img:
                    img.verify()
                with Image.open(path) as img:
                    stats = image_stats(img)
                    w, h = img.size
            except (UnidentifiedImageError, OSError):
                corrupt.append(str(path))
                continue
            records.append({"path": str(path), "label": label, "class_name": cls,
                            "width": w, "height": h, **stats})
    return pd.DataFrame(records), classes, corrupt


def build_manifest(image_dir, manifest_path, val_frac=0.15, test_frac=0.15, seed=42):
    """Create a reproducible stratified train/val/test split and a data-quality report.

    Also writes `reference_stats.json` - the training-set feature distribution that the
    monitoring module later compares live traffic against (data drift detection).
    """
    df, classes, corrupt = scan_images(image_dir)
    df = df.drop_duplicates(subset="path")

    train_df, rest = train_test_split(df, test_size=val_frac + test_frac,
                                      stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                                       stratify=rest["label"], random_state=seed)
    train_df, val_df, test_df = (d.assign(split=s) for d, s in
                                 [(train_df, "train"), (val_df, "val"), (test_df, "test")])
    manifest = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    manifest_path = resolve(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    report = {
        "num_images": int(len(manifest)),
        "num_corrupt_removed": len(corrupt),
        "classes": classes,
        "class_counts": manifest["class_name"].value_counts().sort_index().to_dict(),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "image_sizes": manifest.groupby(["width", "height"]).size().rename("n").reset_index()
                               .to_dict(orient="records"),
        "train_channel_mean": train_df[["r_mean", "g_mean", "b_mean"]].mean().round(4).tolist(),
    }
    with open(manifest_path.parent / "data_report.json", "w") as f:
        json.dump(report, f, indent=2)

    feature_cols = ["brightness", "contrast", "r_mean", "g_mean", "b_mean"]
    reference = {
        "features": {c: train_df[c].round(5).tolist() for c in feature_cols},
        "class_distribution": (train_df["class_name"].value_counts(normalize=True)
                               .reindex(classes).fillna(0).round(5).to_dict()),
    }
    with open(manifest_path.parent / "reference_stats.json", "w") as f:
        json.dump(reference, f)

    return manifest, classes, report


# --------------------------------------------------------------------------- transforms

class RandomRot90:
    """Satellite imagery has no canonical 'up', so 90-degree rotations are label-preserving."""

    def __call__(self, img):
        k = int(torch.randint(0, 4, (1,)))
        return img.rotate(90 * k) if k else img


def get_transforms(img_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            RandomRot90(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------------------------------- datasets

class ManifestDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as img:
            x = self.transform(img.convert("RGB"))
        return x, self.labels[idx]


def load_manifest(cfg):
    path = resolve(cfg["data"]["manifest"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python scripts/prepare_data.py` first.")
    df = pd.read_csv(path)
    classes = (df[["label", "class_name"]].drop_duplicates().sort_values("label")["class_name"].tolist())
    return df, classes


def make_loaders(cfg):
    df, classes = load_manifest(cfg)
    size = cfg["model"]["img_size"]
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    pin = torch.cuda.is_available()

    def loader(split, train):
        ds = ManifestDataset(df[df.split == split], get_transforms(size, train))
        return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=nw, pin_memory=pin,
                          drop_last=train, persistent_workers=nw > 0)

    return loader("train", True), loader("val", False), loader("test", False), classes
