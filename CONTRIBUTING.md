# Contributing to GeoVision

## Set up

```bash
git clone https://github.com/shashankmishra23/geovision-landuse-dl.git
cd geovision-landuse-dl
python -m venv .venv
.venv\Scripts\activate                 # macOS / Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

## Before you push

1. Run the tests — all of them must pass:
   ```bash
   pytest -q
   ```
2. If you changed the web app, start it and click through the page you touched:
   ```bash
   uvicorn app.api:app --port 8000
   ```
3. If you changed the UI, refresh the README images:
   ```bash
   python scripts/capture_screenshots.py
   ```

## Guidelines

- Keep every hyper-parameter in `configs/config.yaml`, not in code.
- Report only numbers you measured; regenerate `artifacts/` with `python -m geovision.evaluate` after retraining.
- Write short, imperative commit messages (e.g. `Add haze drift to the simulator`).
- Open a pull request for larger changes so a teammate can review it.
