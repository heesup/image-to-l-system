#!/usr/bin/env python3
"""
helios_xml_roundtrip_visualizer.py
===================================
Produces a 2-row x 3-column figure:

  Row 1 (BEFORE fix):  Stage 0  ->  Stage 1 (broken: leaf explosion)  ->  Stage 2
  Row 2 (AFTER  fix):  Stage 0  ->  Stage 1 (fixed: MAE=0.0000)       ->  Stage 2

Uses main_old for the "before" row and main for the "after" row.
Both rows share the same Stage-0 XML so the comparison is fair.

Run from repo root:
  PYTHONPATH=. python scratch/helios_xml_roundtrip_visualizer.py
"""

import os
import json
import copy
import socket
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/home/lion397/codes/image-to-l-system")
BUILD_DIR = REPO_ROOT / "Digital-Crops/projects/syntheticdata_generation/build"
MAIN_NEW  = BUILD_DIR / "main_new"   # fixed binary (post-fix)
MAIN_OLD  = BUILD_DIR / "main_old"   # pre-fix binary

SCRATCH = BUILD_DIR / "scratch" / "roundtrip_vis"
SCRATCH.mkdir(parents=True, exist_ok=True)

OUT_FIGURE = REPO_ROOT / "docs/results/assets/fig_helios_xml_roundtrip_vis_stages.png"

REF_PARAMS = BUILD_DIR / "output" / "dap50_gt_0000_params.json"
REF_XML    = BUILD_DIR / "output" / "dap50_gt_0000_plant_0000.xml"


# ---------------------------------------------------------------------------
def get_env() -> dict:
    env = os.environ.copy()
    hostname = socket.gethostname()
    if "gpu" in hostname:
        env["DISPLAY"] = ":1.0"
        print(f"[INFO] GPU host '{hostname}' -> DISPLAY=:1.0")
    return env


# ---------------------------------------------------------------------------
def render_xml_to_vis(binary: Path, xml_path: Path, out_dir: Path, tag: str,
                      ref_params: dict, env: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    params = copy.deepcopy(ref_params)
    params["field"]["plots"][0]["plants"][0]["xml"] = str(xml_path)
    for plot in params["field"]["plots"]:
        for plant in plot["plants"]:
            plant.pop("dap", None)
            plant.pop("archetype", None)

    params_path = out_dir / f"{tag}_params.json"
    params_path.write_text(json.dumps(params, indent=2))

    cmd = [str(binary), "--renderer", "vis",
           "--file", str(params_path),
           "--output", str(out_dir),
           "--name", tag, "--save-xml"]
    print(f"\n[RUN] ({binary.name}) {' '.join(cmd[1:])}")
    result = subprocess.run(cmd, cwd=str(BUILD_DIR), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"{binary.name} failed (rc={result.returncode})")

    candidates = sorted(out_dir.rglob(f"{tag}*_vis.jpeg"))
    if not candidates:
        raise FileNotFoundError(f"No vis image under {out_dir}")
    return candidates[0]


def find_xml(out_dir: Path, tag: str) -> Path:
    candidates = sorted(out_dir.rglob(f"{tag}*_plant_0000.xml"))
    if not candidates:
        raise FileNotFoundError(f"No plant XML under {out_dir} for '{tag}'")
    return candidates[0]


def load_np(p: Path) -> np.ndarray:
    return np.array(
        Image.open(p).convert("RGB").resize((720, 720), Image.LANCZOS),
        dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
def build_panel(rows: list, out_path: Path) -> None:
    TARGET_W, TARGET_H = 720, 720
    HEADER_ROW  = 70
    ROW_LABEL_W = 80
    ARROW_W     = 48
    BG = (18, 18, 18)

    n_cols = len(rows[0][1])
    n_rows = len(rows)
    total_w = ROW_LABEL_W + n_cols * TARGET_W + (n_cols - 1) * ARROW_W
    total_h = n_rows * (HEADER_ROW + TARGET_H)

    canvas = Image.new("RGB", (total_w, total_h), BG)
    draw   = ImageDraw.Draw(canvas)

    try:
        fbig   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        fsub   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        farrow = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        frow   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        fbig = fsub = farrow = frow = ImageFont.load_default()

    ROW_COLORS = [(255, 100, 80), (100, 220, 130)]  # red=before, green=after

    for ri, (row_label, cols) in enumerate(rows):
        y_top     = ri * (HEADER_ROW + TARGET_H)
        row_color = ROW_COLORS[ri % len(ROW_COLORS)]

        # Sidebar
        draw.rectangle([0, y_top, ROW_LABEL_W - 1, y_top + HEADER_ROW + TARGET_H - 1],
                       fill=(28, 28, 28))
        tmp   = Image.new("RGB", (HEADER_ROW + TARGET_H, ROW_LABEL_W), (28, 28, 28))
        tmp_d = ImageDraw.Draw(tmp)
        tw    = tmp_d.textlength(row_label, font=frow)
        tmp_d.text(((HEADER_ROW + TARGET_H - tw) / 2, (ROW_LABEL_W - 14) // 2),
                   row_label, font=frow, fill=row_color)
        canvas.paste(tmp.rotate(90, expand=True), (0, y_top))

        for ci, (col_label, subtitle, img_path) in enumerate(cols):
            x = ROW_LABEL_W + ci * (TARGET_W + ARROW_W)
            img = Image.open(img_path).convert("RGB").resize((TARGET_W, TARGET_H), Image.LANCZOS)
            canvas.paste(img, (x, y_top + HEADER_ROW))

            lw = draw.textlength(col_label, font=fbig)
            draw.text((x + (TARGET_W - lw) / 2, y_top + 6),
                      col_label, font=fbig, fill=row_color)
            if subtitle:
                sw = draw.textlength(subtitle, font=fsub)
                draw.text((x + (TARGET_W - sw) / 2, y_top + 30),
                          subtitle, font=fsub, fill=(200, 200, 200))

            if ci < n_cols - 1:
                ax = x + TARGET_W + ARROW_W // 2 - 8
                ay = y_top + HEADER_ROW + TARGET_H // 2 - 12
                draw.text((ax, ay), "->", font=farrow, fill=(100, 100, 100))

        if ri < n_rows - 1:
            sep_y = y_top + HEADER_ROW + TARGET_H
            draw.line([(0, sep_y), (total_w, sep_y)], fill=(60, 60, 60), width=3)

    canvas.save(str(out_path), quality=95)
    print(f"\n[SAVED] {out_path}")


# ---------------------------------------------------------------------------
def main():
    env = get_env()
    with open(REF_PARAMS) as f:
        ref_params = json.load(f)
    stage0_xml = REF_XML

    # AFTER (fixed)
    print("\n=== AFTER (fixed binary: main) ===")
    s0a_vis = render_xml_to_vis(MAIN_NEW, stage0_xml, SCRATCH/"after_stage0", "after0", ref_params, env)
    s1a_vis = render_xml_to_vis(MAIN_NEW, stage0_xml, SCRATCH/"after_stage1", "after1", ref_params, env)
    s1a_xml = find_xml(SCRATCH/"after_stage1", "after1")
    s2a_vis = render_xml_to_vis(MAIN_NEW, s1a_xml,   SCRATCH/"after_stage2", "after2", ref_params, env)
    mae_a01 = float(np.abs(load_np(s0a_vis) - load_np(s1a_vis)).mean())
    mae_a12 = float(np.abs(load_np(s1a_vis) - load_np(s2a_vis)).mean())
    print(f"[AFTER]  S0<->S1 MAE={mae_a01:.4f}  S1<->S2 MAE={mae_a12:.4f}")

    # BEFORE (broken)
    print("\n=== BEFORE (broken binary: main_old) ===")
    s0b_vis = render_xml_to_vis(MAIN_OLD, stage0_xml, SCRATCH/"before_stage0", "bef0", ref_params, env)
    s1b_vis = render_xml_to_vis(MAIN_OLD, stage0_xml, SCRATCH/"before_stage1", "bef1", ref_params, env)
    s1b_xml = find_xml(SCRATCH/"before_stage1", "bef1")
    s2b_vis = render_xml_to_vis(MAIN_OLD, s1b_xml,   SCRATCH/"before_stage2", "bef2", ref_params, env)
    mae_b01 = float(np.abs(load_np(s0b_vis) - load_np(s1b_vis)).mean())
    mae_b12 = float(np.abs(load_np(s1b_vis) - load_np(s2b_vis)).mean())
    print(f"[BEFORE] S0<->S1 MAE={mae_b01:.4f}  S1<->S2 MAE={mae_b12:.4f}")

    rows = [
        ("BEFORE fix", [
            ("Stage 0  -  Grow (DAP 50)", "Reference plant (seed 42)", s0b_vis),
            ("Stage 1  -  Reload XML -> vis",
             f"MAE vs S0 = {mae_b01:.4f}  (leaf inflation bug)", s1b_vis),
            ("Stage 2  -  Reload again -> vis",
             f"MAE vs S1 = {mae_b12:.4f}", s2b_vis),
        ]),
        ("AFTER fix", [
            ("Stage 0  -  Grow (DAP 50)", "Reference plant (seed 42)", s0a_vis),
            ("Stage 1  -  Reload XML -> vis",
             f"MAE vs S0 = {mae_a01:.4f}  (fixed)", s1a_vis),
            ("Stage 2  -  Reload again -> vis",
             f"MAE vs S1 = {mae_a12:.4f}  (fixed)", s2a_vis),
        ]),
    ]
    build_panel(rows, OUT_FIGURE)

    print(f"\nSummary:")
    print(f"  BEFORE:  S0<->S1 MAE={mae_b01:.4f}  S1<->S2 MAE={mae_b12:.4f}")
    print(f"  AFTER:   S0<->S1 MAE={mae_a01:.4f}   S1<->S2 MAE={mae_a12:.4f}")
    print(f"  Figure:  {OUT_FIGURE}")


if __name__ == "__main__":
    main()
