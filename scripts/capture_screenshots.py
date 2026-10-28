"""Capture README screenshots and an animated demo GIF of the running web app.

Requires the API to be running (uvicorn app.api:app --port 8000) and `pip install playwright`.
Uses the locally installed Microsoft Edge, so no browser download is needed.

Usage:  python scripts/capture_screenshots.py
Output: docs/assets/*.png and docs/assets/demo.gif
"""

import argparse
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"


def wait_idle(page, ms=600):
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(ms)


def frame(page, frames, hold_ms):
    img = Image.open(io.BytesIO(page.screenshot())).convert("RGB")
    frames.append((img, hold_ms))


def run_simulation(page, button):
    page.click(button)
    page.wait_for_function("() => !S.sim && document.querySelector('#simBox').textContent.startsWith('Sent')",
                           timeout=180_000)
    wait_idle(page, 1500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--channel", default="msedge", help="installed browser: msedge or chrome")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gif = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        # pop-up toasts would cover parts of the screenshots
        page.add_init_script("document.addEventListener('DOMContentLoaded', () => { const s = document.createElement('style');"
                             " s.textContent = '#toasts{display:none!important}'; document.head.append(s); });")
        page.on("dialog", lambda d: d.accept())

        # reset monitoring so the screenshots tell a clean story
        page.request.delete(f"{args.url}/monitoring")

        # --- Classify -------------------------------------------------------
        page.goto(f"{args.url}/#classify")
        page.wait_for_function("() => document.querySelectorAll('#samples img').length >= 10 && "
                               "[...document.querySelectorAll('#samples img')].every(i => i.complete)")
        wait_idle(page)
        frame(page, gif, 1800)
        page.click('#samples .sample[data-cls="Industrial"]')
        page.wait_for_function("() => S.last && S.last.label")
        wait_idle(page, 1200)
        page.screenshot(path=OUT / "classify.png", full_page=True)
        page.evaluate("document.querySelector('#resultCard').scrollIntoView({block: 'start'})")
        page.wait_for_timeout(400)
        frame(page, gif, 2600)
        for value in (70, 40):
            page.evaluate(f"(() => {{const o = document.querySelector('#opacity'); o.value = {value}; "
                          "o.dispatchEvent(new Event('input'));})()")
            page.wait_for_timeout(200)
            frame(page, gif, 700)
        page.click('#modeSeg button[data-mode="side"]')
        page.wait_for_timeout(400)
        frame(page, gif, 2200)
        page.locator("#resultCard").screenshot(path=OUT / "classify_result.png")

        # --- Batch ----------------------------------------------------------
        page.goto(f"{args.url}/#batch")
        wait_idle(page)
        page.click("#batchSampleBtn")
        page.wait_for_function("() => S.batch.length >= 20")
        wait_idle(page)
        page.click("#runBatch")
        page.wait_for_function("() => S.batchResults.length >= 20", timeout=120_000)
        wait_idle(page, 1000)
        page.evaluate("window.scrollTo(0, 0)")
        frame(page, gif, 2200)
        page.screenshot(path=OUT / "batch.png", full_page=True)

        # --- Monitoring: normal traffic, then drifted traffic -----------------
        page.goto(f"{args.url}/#monitor")
        wait_idle(page)
        run_simulation(page, "#simNormal")
        page.evaluate("window.scrollTo(0, 0)")
        frame(page, gif, 2200)
        page.screenshot(path=OUT / "monitoring_stable.png", full_page=True)
        run_simulation(page, "#simDrift")
        page.evaluate("window.scrollTo(0, 0)")
        frame(page, gif, 3000)
        page.screenshot(path=OUT / "monitoring_alert.png", full_page=True)

        # --- Model report -----------------------------------------------------
        page.goto(f"{args.url}/#model")
        page.wait_for_function("() => document.querySelectorAll('#gallery img').length > 0")
        page.evaluate("document.querySelectorAll('#gallery img').forEach(i => i.loading = 'eager')")
        page.wait_for_function("() => [...document.querySelectorAll('#gallery img')].every(i => i.complete && i.naturalWidth)")
        wait_idle(page, 800)
        frame(page, gif, 2600)
        page.screenshot(path=OUT / "model_report.png", full_page=True)

        # --- Light theme ------------------------------------------------------
        page.goto(f"{args.url}/#classify")
        page.click("#themeBtn")
        page.wait_for_function("() => document.querySelectorAll('#samples img').length >= 10 && "
                               "[...document.querySelectorAll('#samples img')].every(i => i.complete)")
        page.click('#samples .sample[data-cls="Forest"]')
        page.wait_for_function("() => S.last && S.last.label")
        wait_idle(page, 1200)
        page.evaluate("document.querySelector('#resultCard').scrollIntoView({block: 'start'})")
        page.wait_for_timeout(400)
        page.screenshot(path=OUT / "classify_light.png")
        page.click("#themeBtn")

        page.request.delete(f"{args.url}/monitoring")  # leave a clean log for the user
        browser.close()

    width = 960
    images = [img.resize((width, round(img.height * width / img.width)), Image.LANCZOS) for img, _ in gif]
    images = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
    images[0].save(OUT / "demo.gif", save_all=True, append_images=images[1:], loop=0,
                   duration=[d for _, d in gif], optimize=True)
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1024:8.0f} KB")


if __name__ == "__main__":
    main()
