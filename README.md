# Amazon Inventory Reconciliation Using AI — Counting Items in Bin Images

**UE24CS352A Machine Learning — Mini-Project**
Team: **Tejas Shetty (PES1UG24AM305)** · **Tanishk Gowda (PES1UG24AM301)**

Given a photo of a bin from an Amazon fulfilment centre, predict how many items it holds (0–5) and
check that count against what the inventory system recorded. A disagreement means an item was
probably misplaced, so the bin is flagged for a manual audit.

We reproduce and extend *"Amazon Inventory Reconciliation Using AI"* (Rodriguez Bertorello,
Sripada, Dendumrongsup — Stanford CS229): classical baselines (Logistic Regression, Decision Tree,
SVMs on raw-pixel/PCA, HOG and LoG-blob features) versus ImageNet-pretrained CNNs fine-tuned with
transfer learning (ResNet18/34, EfficientNet-B0).

What we add beyond the paper:
- **Reconciliation, not just counting.** We use the model's probability that the bin holds the
  *recorded* quantity, P(count = recorded), to decide MATCH vs FLAG FOR AUDIT, and evaluate that
  decision directly (ROC-AUC, precision, recall on simulated inventory errors).
- **Count-aware metrics:** RMSE, MAE and off-by-one accuracy alongside accuracy and macro-F1.
- **Modern training recipe:** head warm-up, discriminative learning rates, SGDR / one-cycle,
  AdamW, optional class-weighted loss, mixed precision, test-time augmentation.
- **Explainability:** Grad-CAM in the demo shows which regions drove the prediction.
- **Reproducibility:** one-command pipeline, cached features, fixed seeds, smoke tests.

---

## 1. Setup

Python 3.10–3.12. A GPU is strongly recommended for the CNNs (Google Colab's free T4 is enough).

**Windows (PowerShell)**
```powershell
git clone https://github.com/<user>/amazon-bin-reconciliation.git
cd amazon-bin-reconciliation
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# GPU build of PyTorch (pick your CUDA version at https://pytorch.org/get-started/locally/), e.g.:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m pytest -q tests        # ~1 min, synthetic data, checks everything runs
```

**macOS / Linux**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # on Apple Silicon the MPS backend is used automatically
python -m pytest -q tests
```

**Google Colab:** open `notebooks/colab_pipeline.ipynb`, set the runtime to T4 GPU, run top to bottom.

## 2. Data

The [Amazon Bin Image Dataset](https://registry.opendata.aws/amazon-bin-imagery/) is public on
AWS Open Data (bucket `aft-vbi-pds`), so no AWS account is needed. Each bin has a JPEG and a JSON
whose `EXPECTED_QUANTITY` is our label.

```bash
python -m src.download_data --n 20000 --resize 256 --workers 32   # resumable; ~10-15 min
python -m src.prepare_splits                                      # stratified 70/20/10 + class plot
```
Like the paper, we keep only bins with ≤ 5 items (6 classes, random baseline 16.7 %).
Images are resized to 256×256 on download to save disk. `--balanced` caps each class equally.

No internet? `python -m src.make_synthetic --data-dir data_synth` builds a toy dataset with the same
layout (pipeline testing only — never report numbers from it).

## 3. Running the experiments

```bash
# Classical baselines (CPU). GridSearchCV with stratified k-fold on the train split.
python -m src.train_classical --max-train 6000 --cv 5

# Learning-rate range test (Smith, 2017) -> results/figures/lr_find_resnet34.png
python -m src.train_cnn --arch resnet34 --optimizer sgd --lr-find

# CNNs
python -m src.train_cnn --arch resnet18 --optimizer sgd   --scheduler step     --lr 0.01 --epochs 10 --run-name resnet18_sgd_step
python -m src.train_cnn --arch resnet34 --optimizer sgd   --scheduler sgdr     --lr 0.01 --epochs 12 --run-name resnet34_sgd_sgdr
python -m src.train_cnn --arch resnet34 --optimizer adamw --scheduler onecycle --lr 1e-3 --epochs 12 --class-weights --run-name resnet34_adamw_onecycle_cw
python -m src.train_cnn --arch efficientnet_b0 --optimizer adamw --scheduler onecycle --lr 2e-3 --epochs 12 --run-name effnetb0_adamw_onecycle

# Test-set evaluation (+ TTA), then one comparison table/figure
python -m src.evaluate --ckpt models/resnet34_sgd_sgdr.pt --tta
python -m src.summarize_results
```
Or run everything: `bash scripts/run_all.sh` / `.\scripts\run_all.ps1`.
On Windows add `--num-workers 0` to `train_cnn` / `evaluate` if the DataLoader hangs.

## 4. Live demo

```bash
python app.py --ckpt models/resnet34_sgd_sgdr.pt          # opens http://127.0.0.1:7860
python app.py --ckpt models/resnet34_sgd_sgdr.pt --share  # public link, e.g. from Colab
```
Upload a bin photo (or click a test-set example), enter the recorded quantity, and the app shows
the predicted count, the probability of each count, MATCH / FLAG FOR AUDIT and a Grad-CAM overlay.
CLI equivalent: `python -m src.inference --ckpt <ckpt> --image <jpg> --recorded 3 --cam-out cam.png`.

## 5. Results

`python -m src.summarize_results` writes `results/summary.md` (paste it here after your runs),
`results/summary.csv` and `results/figures/model_comparison.png`. Per-run outputs: training curves
(`curves_*.png`), confusion matrices (`cm_*.png`), most-confident mistakes (`errors_*.png`).

| Model | Test acc | Off-by-one | RMSE | Recon. AUC |
|---|---|---|---|---|
| _fill from results/summary.md_ | | | | |

For context, the paper reports 32 % (SVM + PCA) and 56.2 % / RMSE 0.90 (ResNet34 + Adam, 324K images).
Random guessing: 16.7 %.

## 6. Repository layout

```
app.py                     Gradio demo (count + reconciliation + Grad-CAM)
src/config.py              paths and constants
src/download_data.py       parallel, resumable downloader for the public S3 dataset
src/make_synthetic.py      toy dataset for offline smoke tests
src/prepare_splits.py      stratified 70/20/10 split + class-distribution plot
src/features.py            raw pixels, HOG, HOG-on-edges, LoG blobs, colour histograms (cached)
src/train_classical.py     LogReg / Decision Tree / Linear SVM / RBF SVM with GridSearchCV
src/datasets.py            PyTorch dataset + augmentation
src/models.py              backbones, freezing, discriminative LRs, checkpoint loading
src/engine.py              train loop, prediction with TTA, AMP
src/train_cnn.py           two-phase transfer learning, schedulers, LR finder, early stopping
src/evaluate.py            test metrics, confusion matrix, error gallery
src/metrics.py             accuracy, RMSE, MAE, off-by-one, macro-F1, reconciliation simulation
src/inference.py           single-image prediction, Grad-CAM, MATCH/FLAG decision
src/summarize_results.py   one table + chart from all runs
src/plots.py               plotting helpers
notebooks/colab_pipeline.ipynb  end-to-end run on a Colab GPU
scripts/run_all.{sh,ps1}   full pipeline
tests/test_smoke.py        end-to-end tests on synthetic data
```

## 7. Team contributions

| Member | Responsibilities |
|---|---|
| Tejas Shetty (PES1UG24AM305) | Data pipeline (download, splits, EDA), feature engineering (HOG, LoG blobs, PCA), classical baselines and hyper-parameter search, metrics incl. reconciliation evaluation, write-up |
| Tanishk Gowda (PES1UG24AM301) | CNN transfer-learning pipeline (models, schedulers, LR finder, TTA), GPU experiments, error analysis, Gradio demo with Grad-CAM, slides |

Both: experiment design, result analysis, README, presentation.

## 8. References

1. P. Rodriguez Bertorello, S. Sripada, N. Dendumrongsup. *Amazon Inventory Reconciliation Using AI.* Stanford CS229, 2018.
2. Amazon Bin Image Dataset, AWS Open Data Registry.
3. K. He et al. *Deep Residual Learning for Image Recognition.* CVPR 2016.
4. M. Tan, Q. Le. *EfficientNet.* ICML 2019.
5. I. Loshchilov, F. Hutter. *SGDR: Stochastic Gradient Descent with Warm Restarts.* ICLR 2017.
6. L. N. Smith. *Cyclical Learning Rates for Training Neural Networks.* WACV 2017.
7. R. R. Selvaraju et al. *Grad-CAM.* ICCV 2017.
8. N. Dalal, B. Triggs. *Histograms of Oriented Gradients for Human Detection.* CVPR 2005.
