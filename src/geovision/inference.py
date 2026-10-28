"""Production inference: load a checkpoint once, predict on PIL images, optionally explain."""

import time

import torch
from PIL import Image

from .config import get_device
from .data import get_transforms
from .models import build_model


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]), pretrained=False, dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


class Predictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = torch.device(device) if device else get_device()
        self.model, self.meta = load_checkpoint(checkpoint_path, self.device)
        self.classes = self.meta["classes"]
        self.transform = get_transforms(self.meta["img_size"], train=False)
        self._cam = None

    @property
    def info(self):
        return {
            "model_name": self.meta["model_name"],
            "run_id": self.meta.get("run_id"),
            "version": self.meta.get("version"),
            "created_at": self.meta.get("created_at"),
            "img_size": self.meta["img_size"],
            "classes": self.classes,
            "val_metrics": self.meta.get("val_metrics"),
            "device": self.device.type,
        }

    def _tensor(self, img):
        return self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict_batch(self, images, top_k=3):
        x = torch.cat([self._tensor(img) for img in images])
        probs = self.model(x).softmax(1).cpu()
        return [self._format(p, top_k) for p in probs]

    def predict(self, img, top_k=3, explain=False):
        t0 = time.perf_counter()
        x = self._tensor(img)
        with torch.inference_mode():
            probs = self.model(x).softmax(1)[0].cpu()
        result = self._format(probs, top_k)
        if explain:
            from .explain import GradCAM, overlay_heatmap
            if self._cam is None:
                self._cam = GradCAM(self.model, self.meta["model_name"])
            cam, _ = self._cam(x, class_idx=result["class_index"])
            result["heatmap"] = overlay_heatmap(img, cam)
        result["latency_ms"] = (time.perf_counter() - t0) * 1000
        return result

    def _format(self, probs, top_k):
        k = min(top_k, len(self.classes))
        conf, idx = probs.topk(k)
        return {
            "label": self.classes[int(idx[0])],
            "class_index": int(idx[0]),
            "confidence": float(conf[0]),
            "top_k": [{"label": self.classes[int(i)], "probability": float(c)} for c, i in zip(conf, idx)],
            "probabilities": {c: float(p) for c, p in zip(self.classes, probs)},
        }


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Classify one image from the command line")
    ap.add_argument("image")
    ap.add_argument("--checkpoint", default="artifacts/best_model.pt")
    args = ap.parse_args()
    res = Predictor(args.checkpoint).predict(Image.open(args.image))
    res.pop("probabilities")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
