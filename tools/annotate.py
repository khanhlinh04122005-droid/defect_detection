"""
tools/annotate.py

Tool annotation đơn giản bằng terminal:
    - Duyệt ảnh fail, thêm caption/defect_type/severity/location
    - Lưu JSON annotation cho Stage 3 (VLM fine-tune)
    - Hỗ trợ resume (bỏ qua ảnh đã có annotation)

Cách dùng:
    python tools/annotate.py --category metal_nut
    python tools/annotate.py --category metal_nut --show_image
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from configs.base_config import FAIL_IMAGES_DIR, VLM_ANN_DIR, CATEGORY

console = Console()

DEFECT_TYPES = ["scratch", "dent", "discoloration", "missing", "contamination",
                "crack", "burr", "hole", "stain", "other"]
SEVERITIES   = ["minor", "major", "critical"]
LOCATIONS    = ["top", "bottom", "left", "right", "center",
                "top-left", "top-right", "bottom-left", "bottom-right", "edge", "surface"]


def show_image_terminal(img_path: Path):
    """Hiển thị thông tin ảnh trong terminal (không hiển thị ảnh thực)."""
    try:
        from PIL import Image
        img  = Image.open(img_path)
        w, h = img.size
        console.print(f"[dim]  Kích thước: {w}×{h}px | Mode: {img.mode}[/dim]")
    except Exception:
        pass


def load_existing(ann_path: Path) -> Optional[Dict]:
    if ann_path.exists():
        with open(ann_path, encoding="utf-8") as f:
            return json.load(f)
    return None


def annotate_image(img_path: Path, existing: Optional[Dict] = None) -> Optional[Dict]:
    """
    Hỏi người dùng điền annotation cho 1 ảnh.
    Trả về None nếu skip.
    """
    console.print(Panel(
        f"[bold cyan]{img_path.name}[/bold cyan]",
        title="Annotate",
        subtitle=f"{'[yellow]Đã có annotation[/yellow]' if existing else 'Mới'}",
    ))

    show_image_terminal(img_path)

    if existing:
        console.print(f"  Annotation hiện tại:")
        console.print(f"    caption      : {existing.get('caption', '')}")
        console.print(f"    defect_type  : {existing.get('defect_type', '')}")
        console.print(f"    severity     : {existing.get('severity', '')}")
        console.print(f"    location     : {existing.get('location', '')}")
        if not Confirm.ask("Muốn sửa annotation này?", default=False):
            return existing

    # Skip?
    skip = Prompt.ask(
        "  [s]kip / [q]uit / Enter để tiếp tục",
        default=""
    ).strip().lower()
    if skip == "q":
        return None   # Signal quit
    if skip == "s":
        return "skip"  # Signal skip

    # Defect type
    console.print(f"\n  Loại lỗi: {', '.join(DEFECT_TYPES)}")
    defect_type = Prompt.ask("  defect_type", default=existing.get("defect_type", "") if existing else "scratch")
    if defect_type not in DEFECT_TYPES:
        console.print(f"[yellow]  Dùng '{defect_type}' (không trong danh sách chuẩn)[/yellow]")

    # Severity
    console.print(f"  Mức độ: {', '.join(SEVERITIES)}")
    severity = Prompt.ask("  severity", default=existing.get("severity", "") if existing else "minor")

    # Location
    console.print(f"  Vị trí: {', '.join(LOCATIONS)}")
    location = Prompt.ask("  location", default=existing.get("location", "") if existing else "surface")

    # Caption
    caption = Prompt.ask(
        "  caption (mô tả ngắn bằng tiếng Việt)",
        default=existing.get("caption", "") if existing else ""
    )
    if not caption:
        caption = f"Lỗi {defect_type}, mức {severity}, tại {location}."

    return {
        "caption":      caption,
        "defect_type":  defect_type,
        "severity":     severity,
        "location":     location,
        "pass_fail":    "fail",
        "image_path":   str(img_path),
    }


def run_annotation(category: str, limit: Optional[int] = None):
    img_dir = Path(FAIL_IMAGES_DIR) / category
    ann_dir = Path(VLM_ANN_DIR)
    ann_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))

    if not images:
        console.print(f"[red]Không tìm thấy ảnh trong {img_dir}[/red]")
        return

    if limit:
        images = images[:limit]

    done   = 0
    skipped = 0
    new_ann = 0

    console.print(f"\n[bold]Annotation Tool[/bold] — {category}")
    console.print(f"Tổng ảnh: {len(images)} | Output: {ann_dir}")
    console.print("[dim]Nhập 's' để skip, 'q' để thoát[/dim]\n")

    for i, img_path in enumerate(images, 1):
        ann_path = ann_dir / (img_path.stem + ".json")
        existing = load_existing(ann_path)

        console.print(f"[dim][{i}/{len(images)}][/dim]", end=" ")

        ann = annotate_image(img_path, existing)

        if ann is None:
            console.print("[yellow]Thoát annotation.[/yellow]")
            break
        elif ann == "skip":
            skipped += 1
            console.print(f"[dim]Skip {img_path.name}[/dim]")
            continue
        else:
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=2)
            new_ann += 1 if not existing else 0
            done += 1
            console.print(f"[green]✔ Saved → {ann_path.name}[/green]\n")

    # Summary
    table = Table(title="Annotation Summary", header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Count")
    table.add_row("Annotated (session)", str(done))
    table.add_row("New annotations",     str(new_ann))
    table.add_row("Skipped",             str(skipped))

    all_ann = list(ann_dir.glob("*.json"))
    table.add_row("Total in dir",        str(len(all_ann)))
    console.print(table)


def show_stats(category: str):
    """Hiện thống kê annotation hiện có."""
    from collections import Counter
    ann_dir = Path(VLM_ANN_DIR)
    anns    = list(ann_dir.glob("*.json"))

    if not anns:
        console.print("[yellow]Chưa có annotation nào.[/yellow]")
        return

    types  = Counter()
    sevs   = Counter()
    for f in anns:
        with open(f, encoding="utf-8") as fp:
            d = json.load(fp)
        types[d.get("defect_type", "?")] += 1
        sevs[d.get("severity", "?")]     += 1

    table = Table(title=f"Annotation Stats ({len(anns)} files)", header_style="bold cyan")
    table.add_column("Defect Type")
    table.add_column("Count", justify="right")
    for k, v in types.most_common():
        table.add_row(k, str(v))
    console.print(table)

    table2 = Table(header_style="bold cyan")
    table2.add_column("Severity")
    table2.add_column("Count", justify="right")
    for k, v in sevs.most_common():
        table2.add_row(k, str(v))
    console.print(table2)


def main():
    parser = argparse.ArgumentParser(description="Annotation tool cho VLM fine-tune")
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--limit",    type=int, default=None, help="Số ảnh tối đa mỗi session")
    parser.add_argument("--stats",    action="store_true", help="Chỉ hiện thống kê")
    args = parser.parse_args()

    if args.stats:
        show_stats(args.category)
    else:
        run_annotation(args.category, args.limit)


if __name__ == "__main__":
    main()
