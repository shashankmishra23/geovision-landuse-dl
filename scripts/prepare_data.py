"""Step 1 of the pipeline: download raw data, validate it, split it, and produce EDA figures.

Usage:  python scripts/prepare_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geovision.config import load_config, resolve  # noqa: E402
from geovision.data import build_manifest, download_eurosat  # noqa: E402
from geovision.plots import plot_class_distribution, plot_sample_grid  # noqa: E402


def main():
    cfg = load_config(overrides=sys.argv[1:])
    print("Downloading EuroSAT (~90 MB) ...")
    image_dir = download_eurosat(cfg["data"]["root"])

    print("Validating images and building stratified split ...")
    manifest, classes, report = build_manifest(image_dir, cfg["data"]["manifest"],
                                               cfg["data"]["val_frac"], cfg["data"]["test_frac"], cfg["seed"])
    print(json.dumps({k: report[k] for k in ("num_images", "num_corrupt_removed", "split_counts")}, indent=2))

    out = resolve(cfg["paths"]["artifacts"])
    out.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(report["class_counts"], out / "eda_class_distribution.png")
    plot_sample_grid(manifest, classes, out / "eda_samples.png")
    print(f"Done. Manifest -> {resolve(cfg['data']['manifest'])}, EDA figures -> {out}")


if __name__ == "__main__":
    main()
