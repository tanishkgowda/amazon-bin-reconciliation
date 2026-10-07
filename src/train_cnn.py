"""Fine-tune an ImageNet-pretrained CNN to classify the item count (0..5) of a bin image.

Training recipe (transfer learning):
  1. warm-up: backbone frozen, only the new head is trained (--warmup-epochs)
  2. full fine-tuning with discriminative learning rates (lr/9, lr/3, lr) and a scheduler:
       step     - step decay x0.1 every epochs/3   (paper's "SGD")
       sgdr     - cosine annealing with warm restarts (paper's "SGDR")
       onecycle - one-cycle policy
       cosine   - single cosine decay
  3. best checkpoint (by validation accuracy) is kept; early stopping on --patience.

Examples:
    python -m src.train_cnn --arch resnet34 --optimizer sgd --scheduler sgdr --lr 0.01 --epochs 12
    python -m src.train_cnn --arch resnet34 --optimizer adamw --scheduler onecycle --lr 1e-3
    python -m src.train_cnn --arch resnet34 --lr-find           # LR range test, then exit
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn

from .config import FIG_DIR, MODELS_DIR, NUM_CLASSES, RESULTS_DIR, SEED, data_paths, ensure_dirs
from .datasets import make_loaders
from .engine import get_device, predict, train_one_epoch
from .metrics import compute_metrics
from .models import ARCHS, build_model, param_groups, set_backbone_trainable
from .plots import plot_history, plot_lr_find


def make_optimizer(groups, name, lr, wd):
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9, nesterov=True, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(groups, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    raise ValueError(name)


def make_scheduler(opt, name, epochs, steps_per_epoch, t0):
    """Returns (scheduler, per_batch: bool)."""
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.1), False
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs), False
    if name == "sgdr":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=max(1, t0 * steps_per_epoch), T_mult=2), True
    if name == "onecycle":
        max_lr = [g["lr"] for g in opt.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=epochs * steps_per_epoch,
                                                   pct_start=0.25), True
    if name == "none":
        return None, False
    raise ValueError(name)


def class_weights(train_df, device):
    counts = np.bincount(train_df["quantity"].astype(int), minlength=NUM_CLASSES).astype(float)
    w = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def lr_find(model, loader, criterion, args, device, path, start=1e-7, end=1.0, num_iter=100):
    """Leslie Smith's LR range test: raise lr exponentially, record smoothed loss,
    suggest the lr where loss falls fastest."""
    opt = make_optimizer(model.parameters(), args.optimizer, start, args.weight_decay)
    num_iter = min(num_iter, len(loader) * 3)
    mult = (end / start) ** (1 / num_iter)
    lrs, losses, avg, best, it = [], [], 0.0, math.inf, 0
    model.train()
    while it < num_iter:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            lr = start * mult ** it
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            avg = 0.98 * avg + 0.02 * loss.item()
            smooth = avg / (1 - 0.98 ** (it + 1))
            lrs.append(lr)
            losses.append(smooth)
            best = min(best, smooth)
            it += 1
            if smooth > 4 * best or it >= num_iter:
                break
        else:
            continue
        break
    grads = np.gradient(np.array(losses), np.log(np.array(lrs)))
    lo = max(1, len(grads) // 10)
    suggestion = lrs[lo + int(np.argmin(grads[lo:-1]))] if len(grads) > lo + 1 else lrs[len(lrs) // 2]
    plot_lr_find(lrs, losses, suggestion, path)
    return suggestion


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--arch", default="resnet34", choices=ARCHS)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=12, help="full fine-tuning epochs (after warm-up)")
    p.add_argument("--warmup-epochs", type=int, default=1, help="head-only epochs with frozen backbone")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--optimizer", default="sgd", choices=["sgd", "adam", "adamw"])
    p.add_argument("--scheduler", default="sgdr", choices=["step", "sgdr", "onecycle", "cosine", "none"])
    p.add_argument("--sgdr-t0", type=int, default=1, help="first SGDR cycle length in epochs (doubles each cycle)")
    p.add_argument("--lr", type=float, default=0.01, help="head lr (backbone gets lr/3 and lr/9)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--no-discriminative", action="store_true")
    p.add_argument("--class-weights", action="store_true", help="inverse-frequency weighted cross-entropy")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--max-train", type=int, default=None, help="subsample train split (quick experiments)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--run-name", default=None)
    p.add_argument("--lr-find", action="store_true", help="run the LR range test and exit")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    use_amp = not args.no_amp and device.type == "cuda"
    run = args.run_name or f"{args.arch}_{args.optimizer}_{args.scheduler}"
    ensure_dirs(MODELS_DIR, RESULTS_DIR, FIG_DIR)
    paths = data_paths(args.data_dir)
    loaders, train_df = make_loaders(paths, args.img_size, args.batch_size, args.num_workers, args.max_train, args.seed)
    print(f"[{run}] device={device} amp={use_amp} train={len(train_df)} "
          f"val={len(loaders['val'].dataset)} test={len(loaders['test'].dataset)}")

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device)
    weight = class_weights(train_df, device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    if args.lr_find:
        sug = lr_find(model, loaders["train"], criterion, args, device, FIG_DIR / f"lr_find_{args.arch}.png")
        print(f"Suggested head lr ~ {sug:.2e}  (plot: {FIG_DIR / f'lr_find_{args.arch}.png'})")
        return

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, best_acc, best_epoch, bad = [], -1.0, 0, 0
    ckpt_path = MODELS_DIR / f"{run}.pt"

    def log_epoch(epoch, phase, tr_loss, tr_acc, opt, t0):
        probs, y, val_loss = predict(model, loaders["val"], device, use_amp=use_amp)
        m = compute_metrics(y, probs.argmax(1))
        rec = {"epoch": epoch, "phase": phase, "train_loss": tr_loss, "train_acc": tr_acc,
               "val_loss": val_loss, "val_acc": m["accuracy"], "val_rmse": m["rmse"],
               "lr": max(g["lr"] for g in opt.param_groups), "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"ep {epoch:>2} [{phase}] train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.3f} acc {m['accuracy']:.3f} rmse {m['rmse']:.3f} | {rec['seconds']}s")
        return m["accuracy"]

    def save(epoch, val_acc):
        torch.save({"arch": args.arch, "img_size": args.img_size, "state_dict": model.state_dict(),
                    "epoch": epoch, "val_acc": val_acc, "args": vars(args), "run": run}, ckpt_path)

    epoch = 0
    # Phase 1: train only the new head (random init) so it does not wreck the pretrained features.
    if args.warmup_epochs > 0:
        set_backbone_trainable(model, args.arch, False)
        opt = make_optimizer([q for q in model.parameters() if q.requires_grad], args.optimizer, args.lr,
                             args.weight_decay)
        for _ in range(args.warmup_epochs):
            epoch += 1
            t0 = time.time()
            tl, ta = train_one_epoch(model, loaders["train"], criterion, opt, device, scaler, use_amp)
            acc = log_epoch(epoch, "head", tl, ta, opt, t0)
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                save(epoch, acc)

    # Phase 2: unfreeze everything, discriminative LRs + scheduler.
    set_backbone_trainable(model, args.arch, True)
    groups = param_groups(model, args.arch, args.lr, discriminative=not args.no_discriminative)
    opt = make_optimizer(groups, args.optimizer, args.lr, args.weight_decay)
    sched, per_batch = make_scheduler(opt, args.scheduler, args.epochs, len(loaders["train"]), args.sgdr_t0)
    for _ in range(args.epochs):
        epoch += 1
        t0 = time.time()
        tl, ta = train_one_epoch(model, loaders["train"], criterion, opt, device, scaler, use_amp,
                                 sched if per_batch else None)
        if sched is not None and not per_batch:
            sched.step()
        acc = log_epoch(epoch, "full", tl, ta, opt, t0)
        if acc > best_acc:
            best_acc, best_epoch, bad = acc, epoch, 0
            save(epoch, acc)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping (no val improvement for {args.patience} epochs)")
                break

    with open(RESULTS_DIR / f"{run}_history.json", "w") as f:
        json.dump({"run": run, "args": vars(args), "best_val_acc": best_acc, "best_epoch": best_epoch,
                   "history": history}, f, indent=2)
    plot_history(history, FIG_DIR / f"curves_{run}.png")
    print(f"Best val acc {best_acc:.4f} at epoch {best_epoch}. Checkpoint: {ckpt_path}")
    print(f"Next: python -m src.evaluate --ckpt {ckpt_path} --tta")


if __name__ == "__main__":
    main()
