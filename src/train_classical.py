"""Classical baselines: Logistic Regression, Decision Tree and SVMs on hand-crafted features.

Each experiment = feature set + model. Hyper-parameters are chosen with stratified
k-fold cross-validation on the training split (GridSearchCV); the chosen model is then
scored on the validation and test splits.

    python -m src.train_classical                       # default experiment list
    python -m src.train_classical --max-train 4000 --cv 3 --size 64
    python -m src.train_classical --experiments raw:svm_rbf hog+color+blob:svm_rbf
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import FIG_DIR, RESULTS_DIR, SEED, data_paths, ensure_dirs
from .features import cached_features
from .metrics import compute_metrics
from .plots import plot_confusion

DEFAULT_EXPERIMENTS = [
    "raw:logreg", "raw:tree", "raw:svm_linear", "raw:svm_rbf",
    "hog:svm_rbf", "hog_edge:svm_rbf", "blob:svm_rbf", "hog+color+blob:svm_rbf",
]


def build(model, use_pca, n_pca, class_weight):
    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_pca, svd_solver="randomized", random_state=SEED)))
    if model == "logreg":
        clf, grid = LogisticRegression(max_iter=3000, class_weight=class_weight), {"clf__C": [0.001, 0.01, 0.1, 1]}
    elif model == "tree":
        clf = DecisionTreeClassifier(random_state=SEED, class_weight=class_weight)
        grid = {"clf__max_depth": [5, 10, 20], "clf__min_samples_leaf": [5, 20, 50]}
    elif model == "svm_linear":
        clf, grid = LinearSVC(dual="auto", max_iter=5000, class_weight=class_weight), {"clf__C": [0.0001, 0.001, 0.01, 0.1]}
    elif model == "svm_rbf":
        clf = SVC(kernel="rbf", class_weight=class_weight, cache_size=1000)
        grid = {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.1, 0.01]}
        if use_pca:  # gamma must shrink with dimensionality
            grid["clf__gamma"] = ["scale", 1.0 / (10 * n_pca), 1.0 / (100 * n_pca)]
    else:
        raise ValueError(model)
    steps.append(("clf", clf))
    return Pipeline(steps), grid


def load_feature_set(df, paths, spec, size, n_jobs):
    parts = [cached_features(df, paths["images"], k, size, paths["features"], n_jobs) for k in spec.split("+")]
    return np.hstack(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS, help="feature[+feature]:model")
    p.add_argument("--size", type=int, default=64, help="image side length used for features")
    p.add_argument("--max-train", type=int, default=6000, help="SVMs scale ~O(n^2); subsample the train split")
    p.add_argument("--pca", type=int, default=150, help="PCA components for raw pixels")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--out", default=str(RESULTS_DIR / "classical_results.json"))
    p.add_argument("--fig-dir", default=str(FIG_DIR))
    args = p.parse_args(argv)

    paths = data_paths(args.data_dir)
    ensure_dirs(args.fig_dir, RESULTS_DIR)
    df = pd.read_csv(paths["splits"])
    tr = df[df.split == "train"]
    if len(tr) > args.max_train:  # stratified subsample (keeps the class balance of the split)
        from sklearn.model_selection import train_test_split
        tr, _ = train_test_split(tr, train_size=args.max_train, stratify=tr["quantity"], random_state=SEED)
    va, te = df[df.split == "val"], df[df.split == "test"]
    print(f"train {len(tr)} | val {len(va)} | test {len(te)}")
    cw = None if args.class_weight == "none" else "balanced"

    results = []
    for exp in args.experiments:
        feat, model = exp.split(":")
        t0 = time.time()
        Xtr, Xva, Xte = (load_feature_set(d, paths, feat, args.size, args.n_jobs) for d in (tr, va, te))
        use_pca = feat == "raw"
        n_pca = min(args.pca, Xtr.shape[1], len(Xtr) - 1)
        pipe, grid = build(model, use_pca, n_pca, cw)
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=SEED)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=args.n_jobs)
        gs.fit(Xtr, tr["quantity"].values)
        best = gs.best_estimator_
        probs_te = best.predict_proba(Xte) if hasattr(best, "predict_proba") and model == "logreg" else None
        entry = {
            "name": f"{model} [{feat}]", "feature": feat, "model": model,
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_accuracy": float(gs.best_score_),
            "train_accuracy": float((best.predict(Xtr) == tr["quantity"].values).mean()),
            "val": compute_metrics(va["quantity"].values, best.predict(Xva)),
            "test": compute_metrics(te["quantity"].values, best.predict(Xte), probs_te),
            "n_train": int(len(Xtr)), "seconds": round(time.time() - t0, 1),
        }
        results.append(entry)
        print(f"{entry['name']:<32} cv={entry['cv_accuracy']:.3f} val={entry['val']['accuracy']:.3f} "
              f"test={entry['test']['accuracy']:.3f} rmse={entry['test']['rmse']:.3f} "
              f"params={entry['best_params']} ({entry['seconds']}s)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    best = max(results, key=lambda r: r["val"]["accuracy"])
    plot_confusion(best["test"]["confusion_matrix"], f"{args.fig_dir}/cm_classical_best.png",
                   title=f"Best classical: {best['name']}")
    print(f"Best by validation accuracy: {best['name']} (test acc {best['test']['accuracy']:.3f})")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
