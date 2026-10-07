"""Download a random subset of the Amazon Bin Image Dataset (ABID).

The dataset is public on the AWS Open Data Registry (bucket ``aft-vbi-pds``), so no
AWS account or credentials are needed. Each bin has:
    bin-images/<id>.jpg   - the photo
    metadata/<id>.json    - JSON with EXPECTED_QUANTITY (our label) and per-SKU data

We sample random ids, keep only bins with EXPECTED_QUANTITY <= 5, optionally resize the
image to save disk, and write data/metadata.csv. The script is resumable: re-running it
continues from what is already on disk.

Example:
    python -m src.download_data --n 20000 --resize 256 --workers 32
"""
import argparse
import csv
import io
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm import tqdm

from .config import MAX_QTY, S3_BASE, SEED, TOTAL_IMAGES, data_paths, ensure_dirs

FIELDS = ["image_id", "fname", "quantity", "n_skus"]
_local = threading.local()


def _session(pool):
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.mount("https://", HTTPAdapter(pool_connections=pool, pool_maxsize=pool, max_retries=3))
        _local.s = s
    return _local.s


def fetch_one(idx, img_dir, resize, pool):
    """Fetch metadata (and the image if the label is in range). Returns a dict or None."""
    s = _session(pool)
    try:
        r = s.get(f"{S3_BASE}/metadata/{idx}.json", timeout=20)
        if r.status_code != 200:
            return None
        meta = r.json()
        qty = meta.get("EXPECTED_QUANTITY")
        if qty is None:
            return None
        qty = int(qty)
        if qty > MAX_QTY:
            return {"skip": True, "quantity": qty}
        fname = meta.get("image_fname") or f"{idx}.jpg"
        out = img_dir / fname
        if not out.exists():
            ir = s.get(f"{S3_BASE}/bin-images/{fname}", timeout=30)
            if ir.status_code != 200:
                return None
            img = Image.open(io.BytesIO(ir.content)).convert("RGB")
            if resize:
                img = img.resize((resize, resize), Image.BILINEAR)
            img.save(out, quality=92)
        n_skus = len(meta.get("BIN_FCSKU_DATA") or {})
        return {"image_id": idx, "fname": fname, "quantity": qty, "n_skus": n_skus}
    except Exception:  # network hiccup / corrupt file: just skip this id
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=20000, help="number of usable images (qty<=5) to collect")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--resize", type=int, default=256, help="square resize on save (0 = keep original)")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--balanced", action="store_true",
                   help="cap every class at n/6 images (class 0 is rare, so this needs many more requests)")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    paths = data_paths(args.data_dir)
    ensure_dirs(paths["images"])

    rows, seen = [], set()
    if paths["metadata"].exists():
        with open(paths["metadata"], newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        seen = {int(r["image_id"]) for r in rows}
        print(f"Resuming: {len(rows)} images already in {paths['metadata']}")

    cap = args.n // 6 + 1 if args.balanced else None
    counts = {k: 0 for k in range(MAX_QTY + 1)}
    for r in rows:
        counts[int(r["quantity"])] += 1

    rng = random.Random(args.seed)
    ids = [i for i in range(1, TOTAL_IMAGES + 1) if i not in seen]
    rng.shuffle(ids)

    f = open(paths["metadata"], "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if not rows:
        writer.writeheader()

    pbar = tqdm(total=args.n, initial=len(rows), desc="usable images")
    pos, batch = 0, args.workers * 8
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            while len(rows) < args.n and pos < len(ids):
                chunk = ids[pos:pos + batch]
                pos += batch
                futs = [ex.submit(fetch_one, i, paths["images"], args.resize, args.workers) for i in chunk]
                for fut in as_completed(futs):
                    res = fut.result()
                    if not res or res.get("skip") or len(rows) >= args.n:
                        continue
                    q = res["quantity"]
                    if cap and counts[q] >= cap:
                        continue
                    counts[q] += 1
                    rows.append(res)
                    writer.writerow(res)
                    pbar.update(1)
                f.flush()
    finally:
        f.close()
        pbar.close()

    print("Class counts:", counts)
    print(f"Saved {len(rows)} rows to {paths['metadata']} and images to {paths['images']}")


if __name__ == "__main__":
    main()
