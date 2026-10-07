"""Collect every result file into one comparison table + bar chart for the report/slides.

    python -m src.summarize_results
Produces results/summary.md, results/summary.csv and results/figures/model_comparison.png
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .config import FIG_DIR, RESULTS_DIR, ensure_dirs
from .plots import plot_model_comparison

# Numbers reported by the reference paper (Rodriguez Bertorello et al., Stanford CS229) for context.
PAPER = [
    {"model": "[paper] SVM RBF + PCA (150K imgs)", "accuracy": 0.32, "rmse": None},
    {"model": "[paper] ResNet34 SGDR (150K imgs)", "accuracy": 0.538, "rmse": 0.94},
    {"model": "[paper] ResNet34 Adam (324K imgs)", "accuracy": 0.562, "rmse": 0.90},
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--split", default="test")
    args = p.parse_args(argv)
    rd = Path(args.results_dir)

    rows = []
    cl = rd / "classical_results.json"
    if cl.exists():
        for r in json.load(open(cl)):
            t = r[args.split]
            rows.append({"model": r["name"], "family": "classical", "accuracy": t["accuracy"],
                         "off_by_one": t["off_by_one_accuracy"], "rmse": t["rmse"], "mae": t["mae"],
                         "macro_f1": t["macro_f1"], "recon_auc": None})
    for f in sorted(rd.glob(f"*_{args.split}.json")):
        if f.name.startswith("smoke_test"):
            continue
        m = json.load(open(f))
        rows.append({"model": f.stem.replace(f"_{args.split}", ""), "family": "cnn", "accuracy": m["accuracy"],
                     "off_by_one": m["off_by_one_accuracy"], "rmse": m["rmse"], "mae": m["mae"],
                     "macro_f1": m["macro_f1"], "recon_auc": m.get("reconciliation", {}).get("roc_auc")})
    if not rows:
        print("No results found yet. Run train_classical / train_cnn + evaluate first.")
        return

    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    df.to_csv(rd / "summary.csv", index=False)
    paper = pd.DataFrame(PAPER)
    with open(rd / "summary.md", "w") as fh:
        fh.write(f"## Our results ({args.split} split)\n\n")
        fh.write(df.to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n\n## Reference paper (reported, for context only)\n\n")
        fh.write(paper.to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n\nRandom baseline for 6 classes: 0.167\n")
    ensure_dirs(FIG_DIR)
    plot_model_comparison(df["model"].tolist(), df["accuracy"].tolist(), FIG_DIR / "model_comparison.png")
    print(df.to_string(index=False))
    print(f"\nSaved {rd / 'summary.md'}, {rd / 'summary.csv'}, {FIG_DIR / 'model_comparison.png'}")


if __name__ == "__main__":
    main()
