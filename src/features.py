"""Hand-crafted features for the classical baselines.

  raw     - down-sampled RGB pixels (PCA is applied later inside the sklearn pipeline)
  hog     - Histogram of Oriented Gradients on the grayscale image
  hog_edge- HOG computed on a Sobel edge map (tries to suppress the tape stripes)
  blob    - Laplacian-of-Gaussian blob statistics (count, radii, spatial spread)
  color   - per-channel colour histograms

Features are cached to data/features/<kind>_<size>.npz so repeated runs are fast.
"""
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import blob_log, hog
from skimage.filters import sobel

KINDS = ["raw", "hog", "hog_edge", "blob", "color"]


def load_image(path, size):
    img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _gray(rgb):
    return rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def raw_features(rgb):
    return rgb.reshape(-1)


def hog_features(rgb, edge=False):
    g = _gray(rgb)
    if edge:
        g = sobel(g)
    cell = max(8, rgb.shape[0] // 8)
    return hog(g, orientations=9, pixels_per_cell=(cell, cell), cells_per_block=(2, 2), feature_vector=True)


def blob_features(rgb):
    g = _gray(rgb)
    size = g.shape[0]
    blobs = blob_log(g, min_sigma=size / 40, max_sigma=size / 6, num_sigma=8, threshold=0.08)
    inv = blob_log(1.0 - g, min_sigma=size / 40, max_sigma=size / 6, num_sigma=8, threshold=0.08)
    feats = []
    for b in (blobs, inv):  # bright-on-dark and dark-on-bright blobs
        if len(b) == 0:
            feats += [0] * 11
            continue
        r = b[:, 2] * np.sqrt(2) / size
        hist, _ = np.histogram(r, bins=5, range=(0, 0.25))
        feats += [len(b), r.mean(), r.std(), b[:, 0].std() / size, b[:, 1].std() / size, r.sum(), *hist]
    return np.asarray(feats, dtype=np.float32)


def color_features(rgb, bins=16):
    return np.concatenate([np.histogram(rgb[..., c], bins=bins, range=(0, 1), density=True)[0] for c in range(3)])


def _extract_one(path, kind, size):
    rgb = load_image(path, size)
    if kind == "raw":
        return raw_features(rgb)
    if kind == "hog":
        return hog_features(rgb)
    if kind == "hog_edge":
        return hog_features(rgb, edge=True)
    if kind == "blob":
        return blob_features(rgb)
    if kind == "color":
        return color_features(rgb)
    raise ValueError(kind)


def extract(paths, kind, size=64, n_jobs=-1):
    feats = Parallel(n_jobs=n_jobs, batch_size=64)(delayed(_extract_one)(p, kind, size) for p in paths)
    return np.stack(feats).astype(np.float32)


def cached_features(df, img_dir, kind, size, cache_dir, n_jobs=-1):
    """Features for every row of df (in order), cached by file name."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{kind}_{size}.npz"
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = dict(zip(z["fnames"].tolist(), z["X"]))
    missing = [f for f in df["fname"] if f not in have]
    if missing:
        X_new = extract([img_dir / f for f in missing], kind, size, n_jobs)
        have.update(zip(missing, X_new))
        np.savez_compressed(cache, fnames=np.array(list(have.keys())), X=np.stack(list(have.values())))
    return np.stack([have[f] for f in df["fname"]])
