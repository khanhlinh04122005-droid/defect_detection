"""
stage4_decision/report.py

Tạo báo cáo kiểm tra chất lượng từ kết quả pipeline:
    - JSON report (machine-readable)
    - HTML report (human-readable với ảnh + mask overlay)
    - Console summary (rich)
"""

import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from configs.base_config import RESULTS_DIR

console = Console()

PRIORITY_COLOR = {
    0: "green",
    1: "yellow",
    2: "orange",
    3: "red",
}

PRIORITY_LABEL = {0: "OK", 1: "Minor", 2: "Major", 3: "Critical"}


def save_json_report(result: Dict, output_dir: str = None) -> str:
    """
    Lưu kết quả pipeline thành JSON (bỏ qua numpy arrays).

    Returns:
        đường dẫn file JSON
    """
    out_dir = Path(output_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_stem = Path(result.get("image_path", "unknown")).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"{img_stem}_{timestamp}.json"

    # Tạo bản copy serializable
    clean = _make_serializable(result)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    return str(out_path)


def _make_serializable(obj):
    """Đệ quy loại bỏ numpy arrays và non-serializable objects."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()
                if not isinstance(v, np.ndarray)}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, bool):
        return obj
    return obj


def print_console_report(result: Dict):
    """In báo cáo đẹp ra console với rich."""
    decision = result.get("decision", {})
    verdict  = decision.get("verdict", "Unknown")
    priority = decision.get("priority", 0)
    conf     = decision.get("confidence", 0.0)
    color    = "green" if verdict == "Pass" else "red"

    # Header panel
    console.print(Panel(
        f"[bold {color}]{verdict}[/bold {color}]  "
        f"| Priority: [{PRIORITY_COLOR.get(priority, 'white')}]{PRIORITY_LABEL.get(priority)}[/{PRIORITY_COLOR.get(priority, 'white')}]  "
        f"| Confidence: {conf:.0%}",
        title=f"[bold]Defect Report — {Path(result.get('image_path','')).name}[/bold]",
        subtitle=f"Category: {result.get('category', '')}",
    ))

    # Stage summary table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Stage",    style="bold", width=12)
    table.add_column("Vote",     width=8)
    table.add_column("Detail",   width=50)

    stage_votes = decision.get("stage_votes", {})
    s1 = result.get("stage1", {})
    s2 = result.get("stage2", {})
    s3 = result.get("stage3", {})

    if s1:
        v1 = stage_votes.get("stage1", "-")
        table.add_row(
            "Stage 1 (Anomaly)",
            f"[{'red' if v1=='Fail' else 'green'}]{v1}[/]",
            f"score={s1.get('score',0):.3f}  threshold={s1.get('threshold',0):.3f}  "
            f"({s1.get('elapsed_ms',0)}ms)",
        )

    if s2:
        v2 = stage_votes.get("stage2", "-")
        table.add_row(
            "Stage 2 (Segment)",
            f"[{'red' if v2=='Fail' else 'green'}]{v2}[/]",
            f"area={s2.get('area_ratio',0)*100:.1f}%  "
            f"conf={s2.get('confidence',0):.2f}  "
            f"({s2.get('elapsed_ms',0)}ms)",
        )

    if s3:
        v3 = stage_votes.get("stage3", "-")
        table.add_row(
            "Stage 3 (VLM)",
            f"[{'red' if v3=='Fail' else 'green'}]{v3}[/]",
            f"type={s3.get('defect_type','?')}  "
            f"sev={s3.get('severity','?')}  "
            f"conf={s3.get('confidence',0):.2f}  "
            f"({s3.get('elapsed_ms',0)}ms)",
        )

    console.print(table)

    # VLM caption
    if s3 and s3.get("caption"):
        console.print(f"\n[bold]Mô tả lỗi:[/bold] {s3['caption']}")

    # Reasons
    reasons = decision.get("reasons", [])
    if reasons:
        console.print("\n[bold]Lý do:[/bold]")
        for r in reasons:
            console.print(f"  • {r}")

    if decision.get("needs_review"):
        console.print("\n[bold yellow]⚠ Cần review thủ công[/bold yellow]")

    console.print(f"\n[dim]Tổng thời gian: {result.get('elapsed_ms',0)}ms[/dim]")


def save_html_report(result: Dict, output_dir: str = None) -> str:
    """
    Tạo HTML report với ảnh gốc + mask overlay + bảng kết quả.

    Returns:
        đường dẫn file HTML
    """
    out_dir  = Path(output_dir or RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_stem  = Path(result.get("image_path", "unknown")).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"{img_stem}_{timestamp}.html"

    decision = result.get("decision", {})
    verdict  = decision.get("verdict", "Unknown")
    s1       = result.get("stage1", {})
    s2       = result.get("stage2", {})
    s3       = result.get("stage3", {})

    # Encode ảnh gốc
    img_b64  = _encode_image(result.get("image_path"))

    # Encode overlay (nếu có mask)
    overlay_b64 = ""
    mask_np = s2.get("mask")
    if mask_np is not None and isinstance(mask_np, np.ndarray):
        try:
            img_np  = np.array(Image.open(result["image_path"]).convert("RGB"))
            overlay = _draw_overlay(img_np, mask_np)
            overlay_b64 = _encode_ndarray(overlay)
        except Exception:
            pass

    color  = "#28a745" if verdict == "Pass" else "#dc3545"
    prio   = PRIORITY_LABEL.get(decision.get("priority", 0), "")
    conf   = f"{decision.get('confidence',0):.0%}"
    ts     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = ""
    for stage, key, detail in [
        ("Stage 1", "stage1",
         f"score={s1.get('score',0):.3f} / threshold={s1.get('threshold',0):.3f}"),
        ("Stage 2", "stage2",
         f"mask area={s2.get('area_ratio',0)*100:.1f}% / SAM conf={s2.get('confidence',0):.2f}"),
        ("Stage 3", "stage3",
         f"type={s3.get('defect_type','?')} / severity={s3.get('severity','?')} / {s3.get('caption','')}"),
    ]:
        vote = decision.get("stage_votes", {}).get(key, "-")
        vc   = "#28a745" if vote == "Pass" else "#dc3545"
        rows += f"<tr><td>{stage}</td><td style='color:{vc}'>{vote}</td><td>{detail}</td></tr>"

    reasons_html = "".join(f"<li>{r}</li>" for r in decision.get("reasons", []))

    overlay_section = ""
    if overlay_b64:
        overlay_section = f"""
        <div class="img-box">
            <p><b>Mask Overlay (Stage 2)</b></p>
            <img src="data:image/png;base64,{overlay_b64}" />
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Defect Report — {img_stem}</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; background:#f8f9fa; }}
h1 {{ color: {color}; }}
.verdict-box {{ background:{color}; color:#fff; padding:12px 24px; border-radius:8px; display:inline-block; font-size:1.4em; font-weight:bold; }}
.images {{ display:flex; gap:20px; margin:20px 0; flex-wrap:wrap; }}
.img-box {{ background:#fff; padding:8px; border-radius:6px; box-shadow:0 1px 4px #0002; }}
.img-box img {{ max-width:380px; border-radius:4px; }}
table {{ width:100%; border-collapse:collapse; background:#fff; border-radius:6px; overflow:hidden; box-shadow:0 1px 4px #0002; }}
th {{ background:#343a40; color:#fff; padding:10px 14px; text-align:left; }}
td {{ padding:9px 14px; border-bottom:1px solid #dee2e6; }}
.meta {{ color:#6c757d; font-size:0.9em; margin-top:4px; }}
ul {{ background:#fff; border-radius:6px; padding:12px 24px; box-shadow:0 1px 4px #0002; }}
</style>
</head>
<body>
<h1>Defect Detection Report</h1>
<p class="meta">File: {result.get('image_path','')} | Category: {result.get('category','')} | {ts}</p>

<div class="verdict-box">{verdict}</div>
&nbsp;&nbsp;
<span style="font-size:1.1em">Priority: <b>{prio}</b> | Confidence: <b>{conf}</b>
{"&nbsp;&nbsp;<span style='color:#e67e22'>⚠ Needs Review</span>" if decision.get('needs_review') else ""}</span>

<div class="images">
    <div class="img-box">
        <p><b>Ảnh gốc</b></p>
        <img src="data:image/png;base64,{img_b64}" />
    </div>
    {overlay_section}
</div>

<table>
<tr><th>Stage</th><th>Vote</th><th>Chi tiết</th></tr>
{rows}
</table>

{"<h3>Lý do quyết định</h3><ul>" + reasons_html + "</ul>" if reasons_html else ""}
<p class="meta">Tổng thời gian: {result.get('elapsed_ms',0)}ms</p>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return str(out_path)


def _encode_image(path: str) -> str:
    """Encode ảnh từ file sang base64."""
    try:
        img    = Image.open(path).convert("RGB")
        img    = img.resize((380, 380))
        return _encode_ndarray(np.array(img))
    except Exception:
        return ""


def _encode_ndarray(arr: np.ndarray) -> str:
    """Encode numpy RGB image → base64 PNG string."""
    import io
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _draw_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Vẽ mask đỏ lên ảnh gốc."""
    overlay = image.copy().astype(np.float32)
    red     = np.zeros_like(image, dtype=np.float32)
    red[mask] = [255, 60, 60]
    overlay = overlay * 0.6 + red * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 60, 60), 2)
    return overlay


if __name__ == "__main__":
    print("=== Report smoke test ===")

    fake_result = {
        "image_path": "data/fail/images/metal_nut/001.png",
        "category":   "metal_nut",
        "elapsed_ms": 1234,
        "stage1": {"score": 0.82, "threshold": 0.40, "prediction": "Fail", "elapsed_ms": 210},
        "stage2": {"area_ratio": 0.06, "confidence": 0.79, "elapsed_ms": 380},
        "stage3": {
            "caption": "Vết xước ngang ~4mm ở mặt trên, mức độ major.",
            "defect_type": "scratch", "severity": "major",
            "location": "top", "pass_fail": "Fail",
            "has_defect": True, "confidence": 0.85, "elapsed_ms": 640,
        },
        "decision": {
            "verdict": "Fail", "priority": 2, "priority_label": "Major",
            "confidence": 0.87, "needs_review": False,
            "stage_votes": {"stage1": "Fail", "stage2": "Fail", "stage3": "Fail"},
            "reasons": [
                "Stage1: score=0.820 (2.1× threshold)",
                "Stage2: mask area=6.0% (>2%)",
                "Stage3: VLM=Fail, type=scratch, severity=major, conf=0.85",
            ],
        },
    }

    print_console_report(fake_result)
    json_path = save_json_report(fake_result)
    print(f"\nJSON saved → {json_path}")
    print("OK — HTML report cần ảnh thật để test")
