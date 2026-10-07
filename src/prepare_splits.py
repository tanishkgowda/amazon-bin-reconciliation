"""Stratified 70/20/10 train/val/test split + a class-distribution plot (EDA).

    python -m src.prepare_splits
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import FIG_DIR, SEED, data_paths, ensure_dirs
from .plots import plot_class_distribution


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None)
    p.add_argument("--val", type=float, default=0.2)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--fig-dir", default=str(FIG_DIR))
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args(argv)

    paths = data_paths(args.data_dir)
    df = pd.read_csv(paths["metadata"])
    df = df[df["fname"].apply(lambda f: (paths["images"] / f).exists())].drop_duplicates("fname")
    df["quantity"] = df["quantity"].astype(int)

    train, rest = train_test_split(df, test_size=args.val + args.test,
                                   stratify=df["quantity"], random_state=args.seed)
    val, test = train_test_split(rest, test_size=args.test / (args.val + args.test),
                                 stratify=rest["quantity"], random_state=args.seed)
    train, val, test = train.assign(split="train"), val.assign(split="val"), test.assign(split="test")
    out = pd.concat([train, val, test]).reset_index(drop=True)
    out.to_csv(paths["splits"], index=False)

    print(f"Total {len(out)} | train {len(train)} | val {len(val)} | test {len(test)}")
    print(pd.crosstab(out["quantity"], out["split"]))
    ensure_dirs(args.fig_dir)
    plot_class_distribution(out, f"{args.fig_dir}/class_distribution.png")


if __name__ == "__main__":
    main()
