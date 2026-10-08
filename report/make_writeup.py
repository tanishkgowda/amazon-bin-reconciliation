"""Build the 2-page project write-up PDF.

Results and the numeric parts of the conclusions are filled in automatically from
results/summary.csv and results/*_test.json, so after the experiments just re-run:

    python report/make_writeup.py            -> report/writeup.pdf
Until results exist, the PDF shows clearly marked [placeholders].
"""
import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TEAM = "Tejas Shetty (PES1UG24AM305) &nbsp;&middot;&nbsp; Tanishk Gowda (PES1UG24AM301)"
REPO = "github.com/&lt;user&gt;/amazon-bin-reconciliation (private, shared with faculty)"
ACCENT = colors.HexColor("#1f4e79")

ss = getSampleStyleSheet()
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=15, leading=18, spaceAfter=2, textColor=ACCENT)
SUB = ParagraphStyle("s", parent=ss["Normal"], fontSize=8.8, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
H = ParagraphStyle("h", parent=ss["Heading2"], fontSize=10.5, leading=12.5, spaceBefore=5, spaceAfter=2, textColor=ACCENT)
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.9, leading=11.4, spaceAfter=3)
SMALL = ParagraphStyle("sm", parent=B, fontSize=7.8, leading=9.6, textColor=colors.HexColor("#555555"))


def pct(x):
    return "-" if x is None or pd.isna(x) else f"{100 * x:.1f}%"


def num(x):
    return "-" if x is None or pd.isna(x) else f"{x:.3f}"


def load():
    s = RES / "summary.csv"
    if not s.exists():
        return None, None
    df = pd.read_csv(s)
    best_cnn = df[df.family == "cnn"].sort_values("accuracy", ascending=False).head(1)
    detail = None
    if len(best_cnn):
        f = RES / f"{best_cnn.iloc[0]['model']}_test.json"
        if f.exists():
            detail = json.load(open(f))
    return df, detail


def results_table(df):
    head = ["Model", "Family", "Test acc.", "Off-by-one", "RMSE", "Macro-F1", "Recon. AUC"]
    if df is None:
        rows = [["[run experiments, then re-run this script]", "", "", "", "", "", ""]]
    else:
        rows = [[r.model, r.family, pct(r.accuracy), pct(r.off_by_one), num(r.rmse), num(r.macro_f1),
                 num(r.recon_auc)] for r in df.itertuples()]
    rows.append(["Paper: ResNet34+Adam, 324K imgs (reported)", "ref.", "56.2%", "-", "0.900", "-", "-"])
    rows.append(["Paper: SVM RBF + PCA (reported)", "ref.", "32.0%", "-", "-", "-", "-"])
    rows.append(["Random guess (6 classes)", "ref.", "16.7%", "-", "-", "-", "-"])
    t = Table([head] + rows, colWidths=[62 * mm, 16 * mm, 17 * mm, 18 * mm, 14 * mm, 17 * mm, 19 * mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.6),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
        ("TEXTCOLOR", (0, -3), (-1, -1), colors.HexColor("#666666")),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"), ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0da")),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    return t


def conclusions(df, d):
    if df is None or d is None:
        return [
            "[Auto-filled after experiments: best CNN accuracy vs. best classical model and vs. random; "
            "per-class trend; reconciliation AUC/precision/recall; comparison with the paper.]",
            "Qualitative findings to confirm against your own error gallery: accuracy is highest for empty "
            "bins and falls as the count grows; most errors are off by one; the hardest images have items "
            "hidden behind the elastic tape or stacked on each other, which even a human cannot count reliably.",
        ]
    cnn = df[df.family == "cnn"].iloc[0] if (df.family == "cnn").any() else None
    cl = df[df.family == "classical"].sort_values("accuracy", ascending=False)
    out = []
    if cnn is not None:
        s = (f"The best model, <b>{cnn.model}</b>, reaches <b>{pct(cnn.accuracy)}</b> test accuracy "
             f"({cnn.accuracy / (1 / 6):.1f}x random), {pct(cnn.off_by_one)} off-by-one accuracy and RMSE {num(cnn.rmse)}.")
        if len(cl):
            c = cl.iloc[0]
            s += (f" The best classical model ({c.model}) reaches {pct(c.accuracy)}, so transfer learning gives a "
                  f"{100 * (cnn.accuracy - c.accuracy):.1f}-point gain: learned features beat hand-crafted HOG/blob/PCA "
                  f"features on cluttered, taped bins.")
        out.append(s)
    pc = {int(k): v for k, v in d["per_class_accuracy"].items() if v is not None}
    if pc:
        best_k, worst_k = max(pc, key=pc.get), min(pc, key=pc.get)
        out.append("Per-class test accuracy: " + ", ".join(f"{k} items {pct(v)}" for k, v in sorted(pc.items())) +
                   f". Bins with {best_k} items are easiest and bins with {worst_k} items hardest; middle counts can be "
                   f"confused with neighbours on both sides, and {pct(d.get('off_by_one_accuracy'))} of predictions are "
                   f"within one item of the truth.")
    r = d.get("reconciliation")
    if r:
        out.append(f"For reconciliation, flagging a bin when P(count = recorded) &lt; {r['threshold']} gives ROC-AUC "
                   f"<b>{r['roc_auc']:.3f}</b>, precision {pct(r['precision'])} and recall {pct(r['recall'])} on simulated "
                   f"records with 50% errors. At this threshold the flag fires on most bins, because the model is rarely more "
                   f"than 50% confident in a single count, so the probability is best used to rank bins for audit rather "
                   f"than as an automatic yes/no check.")
    out.append("All CNNs were trained on 14K images, 7-18x fewer than the paper's 105K-260K, which explains most of the "
               "gap to its 54-56%: training accuracy kept rising while validation accuracy plateaued, a sign of too little "
               "data. The CNN variants lie within a few points of each other, close to the noise level of a 2,001-image "
               "test set, and the best run changed optimiser, schedule and class weights together, so its gain cannot be "
               "attributed to a single factor.")
    return out


def build(out_pdf):
    df, d = load()
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=12 * mm, bottomMargin=12 * mm,
                            title="Amazon Inventory Reconciliation Using AI - Write-up", author="Tejas Shetty, Tanishk Gowda")
    S = []
    S.append(Paragraph("Amazon Inventory Reconciliation Using AI: Counting Items in Bin Images", TITLE))
    S.append(Paragraph(f"UE24CS352A Machine Learning, Mini-Project &nbsp;|&nbsp; {TEAM}", SUB))
    S.append(Paragraph(f"Code: {REPO}", SUB))
    S.append(Spacer(1, 4))

    S.append(Paragraph("1. Problem statement", H))
    S.append(Paragraph(
        "In Amazon fulfilment centres, robots carry pods of bins and items are stowed into bins at random. Items are "
        "occasionally misplaced, so the inventory system's recorded count no longer matches what is physically in the "
        "bin. Given only an RGB photo of a bin, we predict the number of items (0 to 5) and use that prediction to "
        "decide whether the bin agrees with its recorded quantity or should be sent for a manual audit. We reproduce "
        "and extend the Stanford CS229 study by Rodriguez Bertorello et al., which compared SVMs with ResNet CNNs.", B))

    S.append(Paragraph("2. Dataset", H))
    S.append(Paragraph(
        "The public <b>Amazon Bin Image Dataset</b> (AWS Open Data, about 535K bin photos) pairs each JPEG with a JSON "
        "whose EXPECTED_QUANTITY field is our label. Following the paper, we keep bins with at most 5 items, giving a "
        "6-class problem with a 16.7% random baseline. Our downloader samples random bins from the public S3 bucket "
        "(20,000 usable images, resized to 256x256), and we use a stratified 70/20/10 train/validation/test "
        "split. The classes are imbalanced (empty bins are rare), photos vary in lighting, and many items are hidden "
        "behind the elastic tape that holds them in place, so some labels cannot be verified even by a human.", B))

    S.append(Paragraph("3. Approach", H))
    S.append(Paragraph(
        "<b>Classical baselines.</b> Images are down-sampled to 64x64 and described by (a) raw pixels reduced with PCA "
        "(150 components, randomized solver), (b) HOG on grayscale and on a Sobel edge map (to suppress tape stripes), "
        "(c) Laplacian-of-Gaussian blob statistics (count, radii, spatial spread of bright and dark blobs) and "
        "(d) colour histograms. We train Logistic Regression, a Decision Tree (Gini), a linear SVM and an RBF-kernel "
        "SVM, with standardisation and class-balanced weights, tuning C, gamma and depth by stratified 5-fold "
        "GridSearchCV on the training split.", B))
    S.append(Paragraph(
        "<b>CNNs with transfer learning.</b> ImageNet-pretrained ResNet18, ResNet34 and EfficientNet-B0 get a new 6-way "
        "head (dropout + linear) trained with cross-entropy loss. Training has two phases: (1) a warm-up epoch with the "
        "backbone frozen so the randomly initialised head does not damage pretrained features, then (2) full fine-tuning "
        "with discriminative learning rates (lr/9 for early layers, lr/3 for middle layers, lr for the head). We compare "
        "SGD with step decay (the paper's baseline), SGD with warm restarts (SGDR), and AdamW with a one-cycle schedule, "
        "with learning rates of 0.01 (SGD) and 1e-3 (AdamW) and a learning-rate range test as a sanity check. Augmentation is kept mild (horizontal flip, "
        "small affine shift, colour jitter), because aggressive crops could cut an item out and silently change the "
        "label. We use optional inverse-frequency class weights, mixed precision, early stopping on validation accuracy, "
        "and flip test-time augmentation.", B))
    S.append(Paragraph(
        "<b>From counting to reconciliation (our extension).</b> The business question is whether a bin matches its "
        "record, not the exact count. We flag a bin when the model's probability for the <i>recorded</i> count, "
        "P(count = recorded), falls below a threshold. To evaluate this, we simulate an inventory system in which 50% of "
        "records are wrong (mostly off by one, as when a single item is misplaced) and report ROC-AUC, precision and "
        "recall of the flag. Since accuracy alone hides how wrong a count is, we also report RMSE, MAE, off-by-one "
        "accuracy and macro-F1.", B))

    S.append(Paragraph("4. Implementation", H))
    S.append(Paragraph(
        "The project is written in Python with scikit-learn and scikit-image for the classical models and PyTorch and "
        "torchvision for the CNNs. Its modules are: a resumable parallel S3 downloader, a stratified splitter with EDA, "
        "a cached feature extractor, GridSearchCV training, a two-phase CNN trainer with an LR finder and four "
        "schedulers, an evaluator that produces a confusion matrix and a gallery of the most confident mistakes, and a "
        "results summariser. A <b>Gradio demo</b> takes a bin photo and the recorded quantity and returns the count "
        "distribution, a MATCH / FLAG FOR AUDIT decision and a <b>Grad-CAM</b> heat-map of the regions behind the "
        "prediction. Seeds are fixed, the whole pipeline runs with one script (PowerShell, bash or a Colab notebook), "
        "and pytest smoke tests on a synthetic dataset check every stage end to end.", B))

    S.append(Paragraph("5. Results (held-out test split)", H))
    S.append(results_table(df))
    S.append(Paragraph("Paper rows are reported numbers on a different sample and split, shown for context only. "
                       "Recon. AUC = mismatch-detection ROC-AUC from P(count = recorded).", SMALL))
    fig = RES / "figures" / "model_comparison.png"
    if fig.exists():
        S.append(Image(str(fig), width=120 * mm, height=60 * mm, kind="proportional"))

    S.append(Paragraph("6. Conclusions", H))
    for c in conclusions(df, d):
        S.append(Paragraph(c, B))
    S.append(Paragraph(
        "<b>Limitations and future work.</b> Labels are noisy when items are fully occluded, and the 0-5 subset "
        "excludes fuller bins. Promising next steps are cleaning or down-weighting unverifiable images, training on "
        "more of the 535K images, ordinal or regression losses that penalise large count errors more, and ensembling "
        "architectures.", B))

    S.append(Paragraph("7. Individual contributions", H))
    S.append(Paragraph(
        "<b>Tejas Shetty:</b> data pipeline and EDA, feature engineering (HOG, LoG blobs, PCA), classical baselines and "
        "hyper-parameter search, evaluation metrics including the reconciliation simulation, write-up. "
        "<b>Tanishk Gowda:</b> CNN transfer-learning pipeline (schedulers, LR finder, TTA), GPU experiments, error "
        "analysis, Gradio demo with Grad-CAM, slides. <b>Both:</b> experiment design, analysis, presentation.", B))
    S.append(Paragraph(
        "References: Rodriguez Bertorello, Sripada, Dendumrongsup, <i>Amazon Inventory Reconciliation Using AI</i> "
        "(Stanford CS229); Amazon Bin Image Dataset (AWS Open Data); He et al., ResNet (CVPR 2016); Tan &amp; Le, "
        "EfficientNet (ICML 2019); Loshchilov &amp; Hutter, SGDR (ICLR 2017); Smith, Cyclical LR (WACV 2017); "
        "Selvaraju et al., Grad-CAM (ICCV 2017); Dalal &amp; Triggs, HOG (CVPR 2005).", SMALL))
    doc.build(S)
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "writeup.pdf")
