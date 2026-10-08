"""Live demo: upload a bin photo + the count the inventory system recorded ->
predicted count, class probabilities, MATCH / FLAG FOR AUDIT decision, Grad-CAM.

    python app.py --ckpt models/resnet34_sgd_sgdr.pt
    python app.py --ckpt models/resnet34_sgd_sgdr.pt --share     # public link (e.g. from Colab)
"""
import argparse
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import data_paths
from src.inference import CountPredictor, reconcile

COLORS = {"MATCH": "#2f855a", "FLAG FOR AUDIT": "#c53030", "NO RECORD": "#4a5568", "OUT OF SCOPE": "#b7791f"}


def build_ui(predictor, examples, threshold):
    def run(img, recorded, show_cam):
        if img is None:
            return None, "Upload a bin image first.", None
        probs = predictor.predict(img)
        rec = None if recorded is None or recorded == "" else int(recorded)
        r = reconcile(probs, rec, threshold)
        md = (f"<h2 style='color:{COLORS[r['status']]}'>{r['status']}</h2>\n\n"
              f"**Predicted items:** {r['predicted']}  (confidence {r['confidence']:.2f})  \n"
              f"**Expected count (probability-weighted):** {r['expected_count']:.2f}  \n")
        if "message" in r:
            md += f"\n{r['message']}"
        cam = predictor.gradcam(img) if show_cam else None
        return {str(k): float(p) for k, p in enumerate(probs)}, md, cam

    with gr.Blocks(title="Amazon Bin Inventory Reconciliation") as demo:
        gr.Markdown("# Amazon Bin Inventory Reconciliation\n"
                    "Count the items in a bin photo (0-5) and check it against the recorded inventory.")
        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Bin image")
                recorded = gr.Number(label="Recorded quantity (from inventory system, optional)", precision=0)
                show_cam = gr.Checkbox(value=True, label="Show Grad-CAM (where the model looked)")
                btn = gr.Button("Check bin", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                probs = gr.Label(num_top_classes=6, label="P(item count)")
                cam = gr.Image(label="Grad-CAM")
        btn.click(run, [img, recorded, show_cam], [probs, verdict, cam])
        if examples:
            gr.Examples(examples, inputs=[img, recorded], label="Test-set examples (true count pre-filled as record)")
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--threshold", type=float, default=0.5, help="flag if P(count == recorded) is below this")
    p.add_argument("--n-examples", type=int, default=8)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    predictor = CountPredictor(args.ckpt)
    examples = []
    paths = data_paths(args.data_dir)
    if Path(paths["splits"]).exists():
        test = pd.read_csv(paths["splits"]).query("split == 'test'")
        if len(test):
            ex = test.groupby("quantity").head(2).head(args.n_examples)
            examples = [[str(paths["images"] / f), int(q)] for f, q in zip(ex["fname"], ex["quantity"])]
    build_ui(predictor, examples, args.threshold).launch(share=args.share)


if __name__ == "__main__":
    main()
