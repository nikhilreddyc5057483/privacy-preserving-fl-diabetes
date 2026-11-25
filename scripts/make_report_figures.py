"""Create tightly cropped report images: highlighted code snippets, notebook cells/outputs
and copies of the key result plots. Output: report_figures/*.png

Run after run_experiments.py and after executing the notebook:
    python scripts/make_report_figures.py
"""
import ast
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_figures"
OUT.mkdir(exist_ok=True)


def trim(im, pad=10, bg=(255, 255, 255)):
    im = im.convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, bg)).getbbox()
    if not box:
        return im
    l, t, r, b = box
    return im.crop((max(l - pad, 0), max(t - pad, 0), min(r + pad, im.width), min(b + pad, im.height)))


# ------------------------------------------------------------------ code snippets
def lines_of(path, name, cls=None, skip_docstring=True):
    """Return (start, end) line numbers of a function (optionally inside a class)."""
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if cls:
        nodes = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
    fn = next(n for n in ast.walk(ast.Module(body=nodes, type_ignores=[]))
              if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    start = fn.lineno
    return start, fn.end_lineno, fn


def code_image(path, start, end, out_name, title, drop=()):
    """Render lines start..end of a file with syntax highlighting and a file/line header."""
    src_lines = path.read_text().splitlines()
    keep = [(i, src_lines[i - 1]) for i in range(start, end + 1) if i not in drop]
    text = "\n".join(l for _, l in keep)
    fmt = ImageFormatter(style="friendly", font_name="DejaVu Sans Mono", font_size=18,
                         line_numbers=True, line_number_start=start, line_number_bg="#eef1f5",
                         line_number_fg="#8a94a6", line_pad=3, image_pad=14)
    png = OUT / out_name
    png.write_bytes(highlight(text, PythonLexer(), fmt))
    code = Image.open(png).convert("RGB")
    # header bar with file name + line range, like an editor tab
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    bar_h = 34
    canvas = Image.new("RGB", (code.width, code.height + bar_h), "white")
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, code.width, bar_h], fill="#1f4e79")
    d.text((12, 7), f"{path.relative_to(ROOT)}  ·  lines {start}–{end}  ·  {title}", fill="white", font=font)
    canvas.paste(code, (0, bar_h))
    d.rectangle([0, 0, canvas.width - 1, canvas.height - 1], outline="#c9d1dc")
    canvas.save(png)
    return png


def docstring_lines(fn):
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant):
        return set(range(fn.body[0].lineno, fn.body[0].end_lineno + 1))
    return set()


def snippet(rel, name, out, title, cls=None, keep_doc=False, start=None, end=None):
    p = ROOT / rel
    s, e, fn = lines_of(p, name, cls)
    drop = set() if keep_doc else docstring_lines(fn)
    return code_image(p, start or s, end or e, out, title, drop)


def find_line(rel, text, after=1):
    for i, l in enumerate((ROOT / rel).read_text().splitlines(), start=1):
        if i >= after and text in l:
            return i
    raise ValueError(text)


def trim_right_editor(im, pad=24):
    """Cut the empty editor background to the right of the longest code line."""
    px = im.load()
    w, h = im.size
    bg = px[w - 20, h // 2]
    last = 0
    for x in range(w - 20, 0, -1):
        if any(abs(sum(px[x, y]) - sum(bg)) > 30 for y in range(8, h - 8)):
            last = x
            break
    right = min(w, last + pad)
    out = im.crop((0, 0, right, h))
    d = ImageDraw.Draw(out)
    d.line([(right - 1, 0), (right - 1, h)], fill="#e0e0e0")
    return out


# ------------------------------------------------------------ notebook screenshots
def notebook_shots(shots):
    nb = ROOT / "notebooks" / "fl_diabetes_experiments.ipynb"
    html = OUT / "_nb.html"
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "html", str(nb),
                    "--output", str(html)], check=True, capture_output=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1150, "height": 900}, device_scale_factor=2)
        page.goto(html.as_uri())
        page.wait_for_timeout(1500)
        page.add_style_tag(content=".jp-OutputPrompt,.jp-InputPrompt{min-width:56px}")
        cells = page.locator(".jp-Cell.jp-CodeCell")
        for idx, part, out in shots:
            target = cells.nth(idx).locator(".jp-InputArea" if part == "in" else ".jp-OutputArea")
            path = OUT / out
            target.screenshot(path=str(path))
            im = trim(Image.open(path), pad=12)
            if part == "in":
                im = trim_right_editor(im)
            im.save(path)
        b.close()
    html.unlink()


def main():
    made = []
    # --- source-code snippets (the algorithm itself) ---
    made.append(snippet("src/data_prep.py", "prepare_data", "code_01_prepare_data.png",
                        "leakage-safe preprocessing"))
    made.append(snippet("src/data_prep.py", "fit", "code_02_median_imputer.png",
                        "zero -> training median", cls="MedianZeroImputer",
                        end=lines_of(ROOT / "src/data_prep.py", "transform", "MedianZeroImputer")[1]))
    made.append(snippet("src/federated.py", "partition_label_skew", "code_03_label_skew.png",
                        "non-IID client shards"))
    made.append(snippet("src/federated.py", "local_sgd", "code_04_local_sgd.png",
                        "local SGD + FedProx term"))
    s = lines_of(ROOT / "src/federated.py", "clip_and_noise")[0]
    e = lines_of(ROOT / "src/federated.py", "fedavg_aggregate")[1]
    made.append(code_image(ROOT / "src/federated.py", s, e, "code_05_dp_and_fedavg.png",
                           "DP-style clipping/noise and FedAvg"))
    s = find_line("src/federated.py", "global_params = LogRegParams.zeros")
    e = find_line("src/federated.py", "history.append", after=s) + 1
    made.append(code_image(ROOT / "src/federated.py", s, e, "code_06_fl_round_loop.png",
                           "one federated round"))
    s = find_line("src/flower_sim.py", "class PimaClient")
    e = find_line("src/flower_sim.py", "return [local.to_vector()]", after=s)
    made.append(code_image(ROOT / "src/flower_sim.py", s, e, "code_07_flower_client.png",
                           "Flower NumPyClient"))
    s = find_line("app/streamlit_app.py", 'st.subheader("1. Consent")')
    e = find_line("app/streamlit_app.py", "return", after=s)
    made.append(code_image(ROOT / "app/streamlit_app.py", s, e, "code_08_consent_gate.png",
                           "consent gate before questionnaire"))
    # --- notebook cells and outputs (code-cell index, part, file) ---
    notebook_shots([
        (2, "out", "nb_01_dataset_summary.png"),
        (4, "in", "nb_02_preprocessing_cell.png"),
        (4, "out", "nb_03_preprocessing_output.png"),
        (5, "out", "nb_04_centralised_results.png"),
        (6, "out", "nb_05_client_shards_table.png"),
        (7, "in", "nb_06_fl_training_cell.png"),
        (8, "out", "nb_07_fl_auc_curve.png"),
        (9, "out", "nb_08_results_comparison.png"),
        (10, "out", "nb_09_dp_sweep.png"),
        (11, "out", "nb_10_crosscheck.png"),
    ])
    # --- key experiment plots (already tight; trimmed again for safety) ---
    for f in ["roc_curves", "fl_auc_per_round", "metric_comparison", "dp_effect",
              "cm_central_lr", "cm_fl5_iid_fedavg", "client_shards", "class_balance",
              "rf_feature_importance"]:
        trim(Image.open(ROOT / "results/figures" / f"{f}.png"), pad=8).save(OUT / f"plot_{f}.png")
    for f in ["01_home", "02_results", "03_privacy", "04_prediction", "06_feedback_questionnaire"]:
        shutil.copy(ROOT / "results/screenshots" / f"{f}.png", OUT / f"app_{f}.png")
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
