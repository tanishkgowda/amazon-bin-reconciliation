"""Evaluate a trained checkpoint on the held-out test split.

Writes results/<run>_test.json, a confusion matrix, and a grid of the most confident
mistakes (useful for error analysis in the report / Q&A).

    python -m src.evaluate --ckpt models/resnet34_sgd_sgdr.pt --tta
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from .config import FIG_DIR, RESULTS_DIR, data_paths, ensure_dirs  # noqa: E402
from .datasets import BinDataset, build_transforms  # noqa: E402
from .engine import get_device, predict  # noqa: E402
from .metrics import compute_metrics  # noqa: E402
from .models import load_checkpoint  # noqa: E402
from .plots import plot_confusion  # noqa: E402


def error_grid(df, img_dir, probs, y, path, k=12):
    pred = probs.argmax(1)
    wrong = (pred != y).nonzero()[0]
    if len(wrong) == 0:
        return
    conf = probs[wrong, pred[wrong]]
    pick = wrong[conf.argsort()[::-1][:k]]
    cols = 4
    rows = (len(pick) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(Image.open(img_dir / df.iloc[i]["fname"]).convert("RGB"))
        ax.set_title(f"true {y[i]} / pred {pred[i]} ({probs[i, pred[i]]:.2f})", fontsize=8)
    fig.suptitle("Most confident mistakes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--tta", action="store_true", help="average predictions with a horizontal flip")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    device = get_device(args.device)
    model, ckpt = load_checkpoint(args.ckpt, device)
    run = ckpt.get("run", Path(args.ckpt).stem)
    paths = data_paths(args.data_dir)
    df = pd.read_csv(paths["splits"])
    df = df[df.split == args.split].reset_index(drop=True)
    loader = DataLoader(BinDataset(df, paths["images"], build_transforms(ckpt["img_size"], False)),
                        batch_size=args.batch_size, num_workers=args.num_workers)
    probs, y, loss = predict(model, loader, device, tta=args.tta, use_amp=device.type == "cuda")
    m = compute_metrics(y, probs.argmax(1), probs)
    m.update({"run": run, "arch": ckpt["arch"], "split": args.split, "tta": args.tta, "loss": loss,
              "best_val_acc_during_training": ckpt.get("val_acc"), "train_args": ckpt.get("args")})

    ensure_dirs(RESULTS_DIR, FIG_DIR)
    tag = f"{run}{'_tta' if args.tta else ''}"
    out = RESULTS_DIR / f"{tag}_{args.split}.json"
    with open(out, "w") as f:
        json.dump(m, f, indent=2, default=str)
    plot_confusion(m["confusion_matrix"], FIG_DIR / f"cm_{tag}.png", title=f"{run} ({args.split})")
    error_grid(df, paths["images"], probs, y, FIG_DIR / f"errors_{tag}.png")

    r = m["reconciliation"]
    print(f"{run} on {args.split} (TTA={args.tta}): acc {m['accuracy']:.4f} | off-by-one {m['off_by_one_accuracy']:.4f} "
          f"| RMSE {m['rmse']:.4f} | MAE {m['mae']:.4f} | macro-F1 {m['macro_f1']:.4f}")
    print("per-class acc:", {k: round(v, 3) if v is not None else None for k, v in m["per_class_accuracy"].items()})
    print(f"reconciliation: AUC {r['roc_auc']:.3f} precision {r['precision']:.3f} recall {r['recall']:.3f}")
    print(f"Saved {out}")


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
