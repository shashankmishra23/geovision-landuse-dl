"""Export the trained model to ONNX (framework-independent serving) and verify numerical parity.

Usage:
    python -m geovision.export --checkpoint artifacts/best_model.pt
"""

import argparse
import json
import time

import numpy as np
import torch

from .config import resolve
from .inference import load_checkpoint


def export_onnx(ckpt_path, out_path):
    model, meta = load_checkpoint(ckpt_path, "cpu")
    size = meta["img_size"]
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(model, dummy, str(out_path), input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"classes": meta["classes"], "img_size": size, "mean": meta["mean"], "std": meta["std"]}, f)
    return model, size


def verify_onnx(model, onnx_path, size, runs=50):
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed - skipping verification")
        return None
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    x = torch.randn(4, 3, size, size)
    with torch.inference_mode():
        ref = model(x).numpy()
    out = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())

    one = x[:1].numpy()
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {"image": one})
    ort_ms = (time.perf_counter() - t0) / runs * 1000
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(runs):
            model(x[:1])
    torch_ms = (time.perf_counter() - t0) / runs * 1000
    return {"max_abs_diff": max_diff, "onnxruntime_ms": ort_ms, "pytorch_cpu_ms": torch_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="artifacts/best_model.pt")
    ap.add_argument("--out", default="artifacts/model.onnx")
    args = ap.parse_args()
    out = resolve(args.out)
    model, size = export_onnx(resolve(args.checkpoint), out)
    print(f"Exported {out} ({out.stat().st_size / 1e6:.1f} MB)")
    check = verify_onnx(model, out, size)
    if check:
        print(f"Parity max|diff|={check['max_abs_diff']:.2e} | "
              f"ONNX Runtime {check['onnxruntime_ms']:.1f} ms vs PyTorch {check['pytorch_cpu_ms']:.1f} ms (CPU, batch 1)")


if __name__ == "__main__":
    main()
