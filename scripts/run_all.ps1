# Full pipeline for Windows PowerShell. Run from the repo root with the venv activated:
#   .\scripts\run_all.ps1            (or)   .\scripts\run_all.ps1 -N 10000
param([int]$N = 20000)
$ErrorActionPreference = "Stop"
python -m src.download_data --n $N --resize 256 --workers 32
python -m src.prepare_splits
python -m src.train_classical --max-train 6000 --cv 5
python -m src.train_cnn --arch resnet18 --optimizer sgd   --scheduler step     --lr 0.01 --epochs 10 --num-workers 0 --run-name resnet18_sgd_step
python -m src.train_cnn --arch resnet34 --optimizer sgd   --scheduler sgdr     --lr 0.01 --epochs 12 --num-workers 0 --run-name resnet34_sgd_sgdr
python -m src.train_cnn --arch resnet34 --optimizer adamw --scheduler onecycle --lr 1e-3 --epochs 12 --num-workers 0 --class-weights --run-name resnet34_adamw_onecycle_cw
python -m src.train_cnn --arch efficientnet_b0 --optimizer adamw --scheduler onecycle --lr 2e-3 --epochs 12 --num-workers 0 --run-name effnetb0_adamw_onecycle
Get-ChildItem models\*.pt | ForEach-Object { python -m src.evaluate --ckpt $_.FullName --tta --num-workers 0 }
python -m src.summarize_results
