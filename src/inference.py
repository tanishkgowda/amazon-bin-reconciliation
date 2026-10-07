"""Single-image inference, Grad-CAM explanation and the reconciliation decision.
Used by the demo app (app.py) and by `python -m src.inference --ckpt ... --image ...`."""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as mcm  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from .config import NUM_CLASSES  # noqa: E402
from .datasets import build_transforms  # noqa: E402
from .engine import get_device  # noqa: E402
from .models import gradcam_layer, load_checkpoint  # noqa: E402


class CountPredictor:
    def __init__(self, ckpt_path, device="auto"):
        self.device = get_device(device)
        self.model, self.ckpt = load_checkpoint(ckpt_path, self.device)
        self.tf = build_transforms(self.ckpt["img_size"], train=False)
        self.target = gradcam_layer(self.model, self.ckpt["arch"])

    def predict(self, img: Image.Image, tta=True):
        x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p = self.model(x).softmax(1)
            if tta:
                p = (p + self.model(torch.flip(x, dims=[3])).softmax(1)) / 2
        return p[0].cpu().numpy()

    def gradcam(self, img: Image.Image, cls=None, alpha=0.45):
        """Grad-CAM heat-map of where the network 'looked' for the predicted count."""
        acts, grads = {}, {}
        def fwd_hook(module, inp, out):
            acts["a"] = out
            out.register_hook(lambda g: grads.__setitem__("g", g))

        h1 = self.target.register_forward_hook(fwd_hook)
        try:
            x = self.tf(img.convert("RGB")).unsqueeze(0).to(self.device)
            self.model.zero_grad()
            out = self.model(x)
            cls = int(out.argmax(1)) if cls is None else cls
            out[0, cls].backward()
            w = grads["g"].mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * acts["a"]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=img.size[::-1], mode="bilinear", align_corners=False)[0, 0]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        finally:
            h1.remove()
        heat = mcm.jet(cam.detach().cpu().numpy())[..., :3]
        base = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return Image.fromarray((255 * ((1 - alpha) * base + alpha * heat)).astype(np.uint8))


def reconcile(probs, recorded=None, threshold=0.5):
    """Decide whether a bin agrees with the recorded inventory count."""
    pred = int(np.argmax(probs))
    expected = float(probs @ np.arange(NUM_CLASSES))
    res = {"predicted": pred, "confidence": float(probs[pred]), "expected_count": expected}
    if recorded is None:
        res["status"] = "NO RECORD"
        return res
    recorded = int(recorded)
    if recorded > NUM_CLASSES - 1:
        res.update(status="OUT OF SCOPE", p_match=None,
                   message=f"Model only covers 0-{NUM_CLASSES - 1} items; send to manual audit.")
        return res
    p_match = float(probs[recorded])
    res["p_match"] = p_match
    if p_match >= threshold:
        res["status"] = "MATCH"
        res["message"] = f"Bin agrees with record ({recorded}); P(match)={p_match:.2f}."
    else:
        res["status"] = "FLAG FOR AUDIT"
        res["message"] = (f"Recorded {recorded} but the model sees {pred} (conf {probs[pred]:.2f}); "
                          f"P(match)={p_match:.2f} < {threshold}.")
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--recorded", type=int, default=None)
    p.add_argument("--cam-out", default=None, help="save a Grad-CAM overlay here")
    args = p.parse_args(argv)
    cp = CountPredictor(args.ckpt)
    img = Image.open(args.image)
    probs = cp.predict(img)
    print("P(count):", {k: round(float(v), 3) for k, v in enumerate(probs)})
    print(reconcile(probs, args.recorded))
    if args.cam_out:
        cp.gradcam(img).save(args.cam_out)
        print("Grad-CAM saved to", args.cam_out)


if __name__ == "__main__":
    main()
