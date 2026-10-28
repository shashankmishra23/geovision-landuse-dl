"""Train every architecture with the same data/seed and print a leaderboard (ablation study).

Usage:  python scripts/compare_models.py --epochs 10
Each model's artifacts are kept in artifacts/<model_name>/; results go to experiments/runs.csv.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from geovision.config import load_config, resolve  # noqa: E402
from geovision.evaluate import evaluate_checkpoint  # noqa: E402
from geovision.train import log_experiment, train  # noqa: E402

RUNS = [
    ("baseline_cnn", ["model.pretrained=false", "model.img_size=64", "train.lr=0.003", "train.warmup_epochs=0"]),
    ("resnet50", ["train.batch_size=32"]),
    ("efficientnet_b0", []),
    ("convnext_tiny", ["train.batch_size=32", "train.lr=0.0005"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--models", nargs="*", default=[m for m, _ in RUNS])
    args = ap.parse_args()

    rows = []
    for name, extra in RUNS:
        if name not in args.models:
            continue
        cfg = load_config(overrides=[f"model.name={name}", f"train.epochs={args.epochs}",
                                     f"paths.artifacts=artifacts/{name}", *extra])
        ckpt, summary = train(cfg)
        test = evaluate_checkpoint(cfg, ckpt)
        log_experiment(cfg, summary, test)
        rows.append({"model": name, "test_acc": test["accuracy"], "test_macro_f1": test["macro_f1"],
                     "ece": test["ece"], "latency_ms": test["latency_ms_per_image"],
                     "train_min": summary["train_minutes"]})

    board = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False)
    print("\n" + board.to_string(index=False, float_format="%.4f"))
    board.to_csv(resolve("experiments/leaderboard.csv"), index=False)


if __name__ == "__main__":
    main()
