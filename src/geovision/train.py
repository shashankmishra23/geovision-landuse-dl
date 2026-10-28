"""Training loop.

Features: two-stage fine-tuning (frozen-backbone warmup -> full fine-tune with discriminative LRs),
mixed precision (AMP), MixUp, label smoothing, cosine LR schedule, gradient clipping,
early stopping on validation macro-F1, best-checkpoint saving, CSV/plot logging,
experiment registry (experiments/runs.csv) and optional MLflow tracking.

Usage:
    python -m geovision.train
    python -m geovision.train --set model.name=resnet50 train.epochs=10
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from . import __version__
from .config import get_device, load_config, resolve, seed_everything
from .data import IMAGENET_MEAN, IMAGENET_STD, make_loaders
from .evaluate import classification_metrics, predict_loader
from .models import build_model, count_parameters, param_groups, set_backbone_trainable
from .plots import plot_history


def mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, use_amp):
    model.train()
    alpha = cfg["train"]["mixup_alpha"]
    max_batches = cfg["train"].get("max_train_batches")
    total_loss, correct, seen = 0.0, 0, 0

    for i, (x, y) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            if alpha > 0:
                x, ya, yb, lam = mixup(x, y, alpha)
                logits = model(x)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            else:
                logits = model(x)
                loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)

    return total_loss / max(seen, 1), correct / max(seen, 1)


def train(cfg):
    seed_everything(cfg["seed"])
    device = get_device()
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    out_dir = resolve(cfg["paths"]["artifacts"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = make_loaders(cfg)
    mcfg, tcfg = cfg["model"], cfg["train"]
    model = build_model(mcfg["name"], len(classes), mcfg["pretrained"], mcfg["dropout"]).to(device)
    print(f"Device={device} | model={mcfg['name']} | params={count_parameters(model)/1e6:.2f}M | "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg["label_smoothing"])
    optimizer = torch.optim.AdamW(param_groups(model, mcfg["name"], tcfg["lr"], tcfg["backbone_lr_mult"]),
                                  weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tcfg["epochs"], eta_min=1e-6)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{mcfg['name']}"
    mlflow = _start_mlflow(cfg, run_id)
    history, best_f1, bad_epochs = [], -1.0, 0
    ckpt_path = out_dir / "best_model.pt"
    t_start = time.time()

    for epoch in range(1, tcfg["epochs"] + 1):
        frozen = mcfg["pretrained"] and mcfg["name"] != "baseline_cnn" and epoch <= tcfg["warmup_epochs"]
        set_backbone_trainable(model, mcfg["name"], not frozen)

        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, cfg, use_amp)
        probs, labels = predict_loader(model, val_loader, device, tta=False)
        val_loss = nn.functional.nll_loss(torch.log(torch.tensor(probs) + 1e-9), torch.tensor(labels)).item()
        val = classification_metrics(probs, labels)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val_loss,
               "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
               "lr": optimizer.param_groups[-1]["lr"], "backbone_frozen": frozen, "time_s": time.time() - t0}
        history.append(row)
        print(f"[{epoch:02d}/{tcfg['epochs']}] loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val['accuracy']:.4f} val_f1 {val['macro_f1']:.4f} "
              f"| {row['time_s']:.0f}s{' (backbone frozen)' if frozen else ''}")
        if mlflow:
            mlflow.log_metrics({k: float(v) for k, v in row.items() if k != "epoch"}, step=epoch)

        if val["macro_f1"] > best_f1:
            best_f1, bad_epochs = val["macro_f1"], 0
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": mcfg["name"],
                "classes": classes,
                "img_size": mcfg["img_size"],
                "dropout": mcfg["dropout"],
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
                "epoch": epoch,
                "val_metrics": val,
                "run_id": run_id,
                "version": __version__,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "config": cfg,
            }, ckpt_path)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch} (no val F1 gain for {bad_epochs} epochs).")
                break

    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot_history(history, out_dir / "training_curves.png")
    print(f"Best val macro-F1 {best_f1:.4f} | total {(time.time()-t_start)/60:.1f} min | saved {ckpt_path}")

    summary = {"run_id": run_id, "model": mcfg["name"], "img_size": mcfg["img_size"],
               "epochs_run": len(history), "best_val_macro_f1": round(best_f1, 4),
               "train_minutes": round((time.time() - t_start) / 60, 2)}
    if mlflow:
        mlflow.log_artifact(str(out_dir / "training_curves.png"))
        mlflow.end_run()
    return ckpt_path, summary


def log_experiment(cfg, summary, test_metrics=None):
    """Append a row to the experiment registry so different models can be compared later."""
    path = resolve(cfg["paths"]["experiments"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(summary)
    if test_metrics:
        row.update({f"test_{k}": round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})
    row["config"] = json.dumps({k: cfg[k] for k in ("model", "train")})
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def _start_mlflow(cfg, run_id):
    try:
        import mlflow
    except ImportError:
        return None
    mlflow.set_experiment(cfg["project_name"])
    mlflow.start_run(run_name=run_id)
    mlflow.log_params({f"{s}.{k}": v for s in ("model", "train") for k, v in cfg[s].items()})
    return mlflow


def main():
    ap = argparse.ArgumentParser(description="Train a land-use classifier")
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5")
    ap.add_argument("--no-eval", action="store_true", help="skip test-set evaluation after training")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    ckpt, summary = train(cfg)
    test_metrics = None
    if not args.no_eval:
        from .evaluate import evaluate_checkpoint
        test_metrics = evaluate_checkpoint(cfg, ckpt)
    log_experiment(cfg, summary, test_metrics)


if __name__ == "__main__":
    main()
