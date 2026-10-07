"""Generate a small synthetic "bin" dataset with the same file layout as the real one.

This is ONLY for checking that the pipeline runs end to end (no internet / no GPU needed).
Never report numbers from it as results. Each image is a brownish bin with 0..5
coloured, partly overlapping boxes plus "tape" stripes and noise.

    python -m src.make_synthetic --n 600 --data-dir data_synth
"""
import argparse
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .config import MAX_QTY, SEED, data_paths, ensure_dirs


def make_image(qty, rng, size=224):
    base = tuple(int(c) for c in rng.integers(110, 160, 3))
    img = Image.new("RGB", (size, size), base)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, size, 18], fill=(70, 60, 50))           # bin rim
    for _ in range(qty):
        w, h = rng.integers(40, 90, 2)
        x, y = rng.integers(5, size - w - 5), rng.integers(25, size - h - 5)
        col = tuple(int(c) for c in rng.integers(40, 240, 3))
        d.rectangle([x, y, x + w, y + h], fill=col, outline=(20, 20, 20), width=2)
    for _ in range(rng.integers(3, 9)):                           # elastic tape across the bin
        ty = int(rng.integers(20, size))
        d.line([0, ty, size, ty + int(rng.integers(-8, 8))], fill=(200, 200, 190), width=2)
    arr = np.asarray(img).astype(np.float32) + rng.normal(0, 12, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--data-dir", default="data_synth")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    paths = data_paths(args.data_dir)
    ensure_dirs(paths["images"])
    rng = np.random.default_rng(args.seed)
    with open(paths["metadata"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "fname", "quantity", "n_skus"])
        w.writeheader()
        for i in range(args.n):
            q = int(i % (MAX_QTY + 1))
            fname = f"synth_{i:05d}.jpg"
            make_image(q, rng).save(paths["images"] / fname, quality=90)
            w.writerow({"image_id": i, "fname": fname, "quantity": q, "n_skus": q})
    print(f"Wrote {args.n} synthetic images to {paths['images']}")


if __name__ == "__main__":
    main()
