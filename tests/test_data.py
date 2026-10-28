import torch
from PIL import Image

from geovision.data import ManifestDataset, get_transforms


def test_manifest_removes_corrupt_and_stratifies(manifest):
    _, df, classes, report = manifest
    assert report["num_corrupt_removed"] == 1
    assert len(df) == 60
    assert classes == ["Forest", "Industrial", "SeaLake"]
    # every class appears in every split
    assert (df.groupby("split")["class_name"].nunique() == 3).all()
    assert set(df.split.unique()) == {"train", "val", "test"}


def test_reference_stats_written(manifest):
    path, *_ = manifest
    ref = path.parent / "reference_stats.json"
    assert ref.exists()


def test_transforms_shape():
    img = Image.new("RGB", (64, 64), (120, 80, 40))
    for train in (True, False):
        x = get_transforms(96, train)(img)
        assert x.shape == (3, 96, 96)
        assert x.dtype == torch.float32


def test_dataset_item(manifest):
    _, df, _, _ = manifest
    ds = ManifestDataset(df[df.split == "train"], get_transforms(64, False))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and isinstance(y, int)
