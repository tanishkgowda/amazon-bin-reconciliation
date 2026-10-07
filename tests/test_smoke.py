"""End-to-end smoke tests on a tiny synthetic dataset (no internet needed).

    python -m pytest -q tests
The CNN test is skipped automatically if PyTorch is not installed.
"""
import importlib.util
import json

import numpy as np
import pytest

from src import make_synthetic, prepare_splits, train_classical
from src.metrics import compute_metrics, simulate_reconciliation


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    make_synthetic.main(["--n", "180", "--data-dir", str(d)])
    prepare_splits.main(["--data-dir", str(d), "--fig-dir", str(d / "figs")])
    return d


def test_metrics_perfect_and_reconciliation():
    y = np.array([0, 1, 2, 3, 4, 5] * 20)
    m = compute_metrics(y, y, np.eye(6)[y])
    assert m["accuracy"] == 1.0 and m["rmse"] == 0.0
    assert m["reconciliation"]["roc_auc"] == 1.0
    uniform = simulate_reconciliation(np.full((120, 6), 1 / 6), y)
    assert 0.3 < uniform["roc_auc"] < 0.7          # an uninformative model cannot detect mismatches


def test_classical_pipeline(synth, tmp_path):
    out = tmp_path / "res.json"
    train_classical.main(["--data-dir", str(synth), "--experiments", "hog:svm_rbf", "blob:svm_rbf",
                          "--cv", "2", "--out", str(out), "--fig-dir", str(tmp_path), "--n-jobs", "1"])
    res = json.load(open(out))
    assert len(res) == 2 and all(0 <= r["test"]["accuracy"] <= 1 for r in res)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch not installed")
def test_cnn_pipeline(synth):
    from src import evaluate, train_cnn
    from src.config import MODELS_DIR
    from src.inference import CountPredictor, reconcile
    from PIL import Image

    train_cnn.main(["--data-dir", str(synth), "--arch", "resnet18", "--no-pretrained", "--img-size", "64",
                    "--epochs", "1", "--warmup-epochs", "1", "--batch-size", "16", "--num-workers", "0",
                    "--run-name", "smoke_test", "--device", "cpu"])
    ckpt = MODELS_DIR / "smoke_test.pt"
    assert ckpt.exists()
    evaluate.main(["--ckpt", str(ckpt), "--data-dir", str(synth), "--tta", "--num-workers", "0", "--device", "cpu"])
    cp = CountPredictor(ckpt, device="cpu")
    img = Image.open(next((synth / "images").glob("*.jpg")))
    probs = cp.predict(img)
    assert abs(probs.sum() - 1) < 1e-4
    assert reconcile(probs, 2)["status"] in {"MATCH", "FLAG FOR AUDIT"}
    assert cp.gradcam(img).size == img.size
    # clean up so smoke-test artefacts never end up in the results summary
    from src.config import FIG_DIR, RESULTS_DIR
    for f in [ckpt, *RESULTS_DIR.glob("smoke_test*"), *FIG_DIR.glob("*smoke_test*")]:
        f.unlink(missing_ok=True)
