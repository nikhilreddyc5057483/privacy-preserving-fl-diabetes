"""Capture a screenshot of every Streamlit tab with headless Chromium (Playwright).

Start the app first:  streamlit run app/streamlit_app.py --server.headless true --server.port 8501
Then:                 python scripts/capture_screenshots.py [--url http://localhost:8501]
Output:               results/screenshots/<tab>.png (trailing white space trimmed)
"""
import argparse
from pathlib import Path

from PIL import Image, ImageChops
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parents[1] / "results" / "screenshots"


def trim(path: Path, pad: int = 16):
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    if box:
        l, t, r, b = box
        im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width),
                 min(b + pad, im.height))).save(path)


def main(url):
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 2600}, device_scale_factor=1)
        page.goto(url, wait_until="networkidle")
        page.get_by_text("Privacy-Preserving Diabetes Prediction").first.wait_for(timeout=30000)
        page.wait_for_timeout(2500)

        def tab(name):
            page.get_by_role("tab", name=name).click()
            page.wait_for_timeout(2500)

        def shot(name):
            page.wait_for_timeout(1200)
            path = OUT / f"{name}.png"
            page.screenshot(path=str(path))
            trim(path)
            print("saved", path)

        shot("01_home")
        tab("📊 Results"); shot("02_results")
        tab("🛡️ Privacy explained"); shot("03_privacy")
        tab("🧪 Try a prediction")
        page.get_by_role("button", name="Get risk score").click()
        page.wait_for_timeout(2000); shot("04_prediction")
        tab("📝 Feedback"); shot("05_feedback_consent_gate")
        for cb in page.locator("[data-testid='stCheckbox'] label").all():
            cb.click(); page.wait_for_timeout(300)
        page.wait_for_timeout(1500); shot("06_feedback_questionnaire")
        browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    main(ap.parse_args().url)
