"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

MAX_QTY = 5                      # we keep bins holding 0..5 items (same scope as the reference paper)
NUM_CLASSES = MAX_QTY + 1        # classes 0,1,2,3,4,5
CLASS_NAMES = [str(i) for i in range(NUM_CLASSES)]
SEED = 42

# Amazon Bin Image Dataset (public, AWS Open Data Registry, bucket "aft-vbi-pds")
S3_BASE = "https://aft-vbi-pds.s3.amazonaws.com"
TOTAL_IMAGES = 535234

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def data_paths(data_dir=None):
    """Return the standard file locations inside a data directory."""
    d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return {
        "root": d,
        "images": d / "images",
        "metadata": d / "metadata.csv",
        "splits": d / "splits.csv",
        "features": d / "features",
    }


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
