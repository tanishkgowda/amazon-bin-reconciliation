"""Plotting helpers (matplotlib only, saved as PNG for the write-up and slides)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import CLASS_NAMES  # noqa: E402


def plot_class_distribution(df, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = df.groupby(["quantity", "split"]).size().unstack(fill_value=0)
    counts = counts[[c for c in ["train", "val", "test"] if c in counts]]
    counts.plot(kind="bar", stacked=True, ax=ax, color=["#2b6cb0", "#ed8936", "#718096"])
    ax.set_xlabel("Items in bin")
    ax.set_ylabel("Images")
    ax.set_title("Class distribution")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, title="Confusion matrix", normalize=True):
    cm = np.asarray(cm, dtype=float)
    shown = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted count")
    ax.set_ylabel("True count")
    ax.set_title(title)
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            txt = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > shown.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history, path):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train")
    axes[1].plot(ep, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    for a in axes:
        a.set_xlabel("Epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_model_comparison(names, accs, path, baseline=1 / 6):
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(names) + 1.2))
    order = np.argsort(accs)
    ax.barh(np.array(names)[order], np.array(accs)[order] * 100, color="#2b6cb0")
    ax.axvline(baseline * 100, color="#c53030", ls="--", label=f"random ({baseline*100:.1f}%)")
    for i, v in enumerate(np.array(accs)[order]):
        ax.text(v * 100 + 0.5, i, f"{v*100:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Test accuracy (%)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lr_find(lrs, losses, suggestion, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(lrs, losses)
    ax.axvline(suggestion, color="#c53030", ls="--", label=f"suggested lr={suggestion:.2e}")
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate (log)")
    ax.set_ylabel("Smoothed loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
