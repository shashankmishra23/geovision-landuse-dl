"""Send test images to the running API to demo monitoring - optionally with synthetic drift.

    python scripts/simulate_traffic.py --n 200                 # normal traffic  -> drift status "ok"
    python scripts/simulate_traffic.py --n 200 --drift dark    # darker/hazy images -> "warning"/"alert"
"""

import argparse
import io
import random
import sys
from pathlib import Path

import requests
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from geovision.config import load_config  # noqa: E402
from geovision.data import load_manifest  # noqa: E402

DRIFTS = {
    "none": lambda im: im,
    "dark": lambda im: ImageEnhance.Brightness(im).enhance(0.45),          # e.g. a new sensor / dusk imagery
    "haze": lambda im: ImageEnhance.Contrast(im).enhance(0.4).filter(ImageFilter.GaussianBlur(1.5)),
    "color": lambda im: Image.merge("RGB", im.split()[::-1]),              # wrong band order (BGR)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--drift", choices=DRIFTS, default="none")
    args = ap.parse_args()

    df, _ = load_manifest(load_config())
    sample = df[df.split == "test"].sample(args.n, random_state=random.randint(0, 9999))
    for i, (_, r) in enumerate(sample.iterrows(), 1):
        img = DRIFTS[args.drift](Image.open(r.path).convert("RGB"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        res = requests.post(f"{args.api}/predict", files={"file": (Path(r.path).name, buf.getvalue(), "image/jpeg")})
        res.raise_for_status()
        if random.random() < 0.3:  # analysts label ~30% of traffic
            requests.post(f"{args.api}/feedback", json={"request_id": res.json()["request_id"],
                                                        "true_label": r.class_name})
        if i % 50 == 0:
            print(f"sent {i}/{args.n}")
    print(requests.get(f"{args.api}/monitoring/drift").json()["status"])


if __name__ == "__main__":
    main()
