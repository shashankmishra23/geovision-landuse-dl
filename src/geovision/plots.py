"""All figures used in the report: EDA, training curves, confusion matrix, calibration, errors."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_class_distribution(counts, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    names, values = list(counts.keys()), list(counts.values())
    ax.barh(names, values, color="#3b6ea5")
    ax.set_xlabel("images")
    ax.set_title("Class distribution")
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", fontsize=8)
    _save(fig, path)


def plot_sample_grid(df, classes, path, per_class=5, seed=0):
    fig, axes = plt.subplots(len(classes), per_class, figsize=(per_class * 1.6, len(classes) * 1.6))
    for r, cls in enumerate(classes):
        sample = df[df.class_name == cls].sample(per_class, random_state=seed)
        for c, p in enumerate(sample.path):
            ax = axes[r, c]
            ax.imshow(Image.open(p).convert("RGB"))
            ax.axis("off")
            if c == 0:
                ax.set_title(cls, fontsize=8, loc="left")
    _save(fig, path)


def plot_history(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(epochs, [h["train_loss"] for h in history], label="train")
    a1.plot(epochs, [h["val_loss"] for h in history], label="val")
    a1.set_title("Loss")
    a2.plot(epochs, [h["train_acc"] for h in history], label="train acc")
    a2.plot(epochs, [h["val_acc"] for h in history], label="val acc")
    a2.plot(epochs, [h["val_macro_f1"] for h in history], label="val macro-F1", linestyle="--")
    a2.set_title("Accuracy / F1")
    for a in (a1, a2):
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend()
    _save(fig, path)


def plot_confusion_matrix(cm, classes, path):
    cm_norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_reliability(probs, labels, path, n_bins=15):
    conf, correct = probs.max(1), (probs.argmax(1) == labels)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            centers.append(conf[m].mean())
            accs.append(correct[m].mean())
            counts.append(m.sum())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(centers, accs, "o-", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, path)


def plot_misclassified(paths, labels, probs, classes, path, n=16):
    pred = probs.argmax(1)
    wrong = np.where(pred != labels)[0]
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-probs[wrong].max(1))][:n]  # most confident mistakes first
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.set_title(f"true: {classes[labels[i]]}\npred: {classes[pred[i]]} ({probs[i].max():.2f})", fontsize=7)
    fig.suptitle("Most confident mistakes")
    _save(fig, path)
