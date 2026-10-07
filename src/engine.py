"""Device selection, one training epoch, and batched prediction (with optional TTA)."""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F


def get_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, batch_scheduler=None):
    model.train()
    tot_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, use_amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if batch_scheduler is not None:
            batch_scheduler.step()
        tot_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device, tta=False, use_amp=False):
    """Return (probs [N,C], labels [N], mean CE loss). TTA = average with horizontal flip."""
    model.eval()
    probs, labels, tot_loss, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, use_amp):
            logits = model(x).float()
            p = logits.softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).float().softmax(1)) / 2
        tot_loss += F.nll_loss(torch.log(p.clamp_min(1e-8)), y, reduction="sum").item()
        n += len(y)
        probs.append(p.cpu().numpy())
        labels.append(y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(labels), tot_loss / max(n, 1)
