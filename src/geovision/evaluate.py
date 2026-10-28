"""Held-out test evaluation.

Reports accuracy, macro/weighted F1, top-3 accuracy, calibration (ECE), per-class metrics,
confusion matrix, reliability diagram, misclassified examples and inference latency.

Usage:
    python -m geovision.evaluate --checkpoint artifacts/best_model.pt
"""

import argparse
import json
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import get_device, load_config, resolve
from .data import load_manifest, make_loaders
from .plots import plot_confusion_matrix, plot_misclassified, plot_reliability


@torch.inference_mode()
def predict_loader(model, loader, device, tta=False):
    """Return (probs [N, C], labels [N]). TTA averages original + horizontal + vertical flips."""
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        p = model(x).float().softmax(1)
        if tta:
            p = (p + model(x.flip(3)).float().softmax(1) + model(x.flip(2)).float().softmax(1)) / 3
        all_probs.append(p.cpu())
        all_labels.append(y)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def expected_calibration_error(probs, labels, n_bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(probs, labels):
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {
        "accuracy": float((pred == labels).mean()),
        "macro_f1": float(f1_score(labels, pred, average="macro")),
        "weighted_f1": float(f1_score(labels, pred, average="weighted")),
        "top3_accuracy": float((top3 == labels[:, None]).any(1).mean()),
        "ece": expected_calibration_error(probs, labels),
        "mean_confidence": float(probs.max(1).mean()),
    }


@torch.inference_mode()
def benchmark_latency(model, img_size, device, runs=50):
    model.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(10):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def evaluate_checkpoint(cfg, ckpt_path, tta=None):
    from .inference import load_checkpoint

    device = get_device()
    model, meta = load_checkpoint(ckpt_path, device)
    cfg["model"]["img_size"] = meta["img_size"]
    _, _, test_loader, classes = make_loaders(cfg)
    tta = cfg["eval"]["tta"] if tta is None else tta

    probs, labels = predict_loader(model, test_loader, device, tta=tta)
    metrics = classification_metrics(probs, labels)
    metrics["tta"] = tta
    metrics["latency_ms_per_image"] = benchmark_latency(model, meta["img_size"], device)
    metrics["device"] = device.type
    metrics["num_test_images"] = int(len(labels))

    out = resolve(cfg["paths"]["artifacts"])
    pred = probs.argmax(1)
    report = classification_report(labels, pred, target_names=classes, digits=4)
    (out / "classification_report.txt").write_text(report)
    metrics["per_class"] = classification_report(labels, pred, target_names=classes, output_dict=True)
    with open(out / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(confusion_matrix(labels, pred), classes, out / "confusion_matrix.png")
    plot_reliability(probs, labels, out / "reliability_diagram.png")
    df, _ = load_manifest(cfg)
    test_paths = df[df.split == "test"]["path"].tolist()
    plot_misclassified(test_paths, labels, probs, classes, out / "misclassified.png")

    print(report)
    print(f"Test accuracy {metrics['accuracy']:.4f} | macro-F1 {metrics['macro_f1']:.4f} | "
          f"top-3 {metrics['top3_accuracy']:.4f} | ECE {metrics['ece']:.4f} | "
          f"{metrics['latency_ms_per_image']:.1f} ms/img on {device.type}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on the held-out test set")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="artifacts/best_model.pt")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    evaluate_checkpoint(cfg, resolve(args.checkpoint), tta=False if args.no_tta else None)


if __name__ == "__main__":
    main()
