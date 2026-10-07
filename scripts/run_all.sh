#!/usr/bin/env bash
# Full pipeline (Linux / macOS / Colab). Edit N for dataset size.
set -e
N=${N:-20000}
python -m src.download_data --n "$N" --resize 256 --workers 32
python -m src.prepare_splits
python -m src.train_classical --max-train 6000 --cv 5
python -m src.train_cnn --arch resnet18 --optimizer sgd   --scheduler step     --lr 0.01  --epochs 10 --run-name resnet18_sgd_step
python -m src.train_cnn --arch resnet34 --optimizer sgd   --scheduler sgdr     --lr 0.01  --epochs 12 --run-name resnet34_sgd_sgdr
python -m src.train_cnn --arch resnet34 --optimizer adamw --scheduler onecycle --lr 1e-3  --epochs 12 --class-weights --run-name resnet34_adamw_onecycle_cw
python -m src.train_cnn --arch efficientnet_b0 --optimizer adamw --scheduler onecycle --lr 2e-3 --epochs 12 --run-name effnetb0_adamw_onecycle
for c in models/*.pt; do python -m src.evaluate --ckpt "$c" --tta; done
python -m src.summarize_results
