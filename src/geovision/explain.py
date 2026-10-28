"""Explainability with Grad-CAM: which pixels drove the prediction?

Usage:
    python -m geovision.explain --checkpoint artifacts/best_model.pt --n 12
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from .config import load_config, resolve  # noqa: E402
from .models import cam_layer  # noqa: E402


class GradCAM:
    def __init__(self, model, model_name):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = cam_layer(model, model_name)
        self._handle = layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        if not output.requires_grad:  # plain inference call - nothing to record
            return
        self.activations = output
        output.register_hook(lambda g: setattr(self, "gradients", g))

    def __call__(self, x, class_idx=None):
        """x: [1, 3, H, W] normalized tensor. Returns (cam [H, W] in 0..1, class_idx)."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / cam.max().clamp(min=1e-8)
        return cam.detach().cpu().numpy(), class_idx

    def remove(self):
        self._handle.remove()


def overlay_heatmap(img, cam, alpha=0.45):
    """Blend a [H, W] CAM over a PIL image and return a PIL image of the same size."""
    img = img.convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BILINEAR)
    heat = plt.get_cmap("jet")(np.asarray(cam_img) / 255.0)[..., :3]
    blended = (1 - alpha) * (np.asarray(img) / 255.0) + alpha * heat
    return Image.fromarray(np.uint8(blended.clip(0, 1) * 255))


def main():
    from .data import load_manifest
    from .inference import Predictor

    ap = argparse.ArgumentParser(description="Generate a Grad-CAM gallery from test images")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", default="artifacts/best_model.pt")
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config(args.config)
    predictor = Predictor(resolve(args.checkpoint))
    df, _ = load_manifest(cfg)
    sample = df[df.split == "test"].groupby("class_name").sample(1, random_state=cfg["seed"]).head(args.n)

    cols = 4
    rows = int(np.ceil(len(sample) / cols))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, (_, r) in enumerate(sample.iterrows()):
        img = Image.open(r.path).convert("RGB")
        res = predictor.predict(img, explain=True)
        ax_img, ax_cam = axes[k // cols, (k % cols) * 2], axes[k // cols, (k % cols) * 2 + 1]
        ax_img.imshow(img)
        ax_img.set_title(f"true: {r.class_name}", fontsize=7)
        ax_cam.imshow(res["heatmap"])
        ax_cam.set_title(f"pred: {res['label']} ({res['confidence']:.2f})", fontsize=7)
    fig.suptitle("Grad-CAM: regions driving each prediction")
    fig.tight_layout()
    out = resolve(cfg["paths"]["artifacts"]) / "gradcam_examples.png"
    fig.savefig(out, dpi=130)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
