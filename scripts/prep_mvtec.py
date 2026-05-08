#!/usr/bin/env python3
"""
scripts/prepare_mvtec.py

Convert MVTec dataset → project format
"""

import argparse
import shutil
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# CONSTANTS

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

RECOMMENDED = ["metal_nut", "tile", "leather", "grid", "carpet"]


# UTILS

def transfer_file(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "symlink":
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def list_images(folder: Path) -> List[Path]:
    return sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))


def validate_category(root: Path, category: str) -> bool:
    required = [
        root / category / "train" / "good",
        root / category / "test",
        root / category / "ground_truth",
    ]
    for p in required:
        if not p.exists():
            console.print(f"[red]Missing:[/red] {p}")
            return False
    return True


def get_defect_types(root: Path, category: str) -> List[str]:
    test_dir = root / category / "test"
    return [
        d.name for d in test_dir.iterdir()
        if d.is_dir() and d.name != "good"
    ]


# CORE LOGIC

def process_pass_images(src_dir: Path, dst_dir: Path, prefix: str, stats: Dict, mode: str, key: str):
    for img in list_images(src_dir):
        dst = dst_dir / f"{prefix}_{img.name}"
        transfer_file(img, dst, mode)
        stats[key] += 1


def process_fail_images(
    root: Path,
    category: str,
    defect_types: List[str],
    output_dir: Path,
    stats: Dict,
    mode: str
):
    for defect in defect_types:
        test_dir = root / category / "test" / defect
        gt_dir   = root / category / "ground_truth" / defect

        for img in list_images(test_dir):
            stem = img.stem
            name = f"{category}_{defect}_{img.name}"

            # image
            dst_img = output_dir / "fail/images" / name
            transfer_file(img, dst_img, mode)
            stats["fail_images"] += 1

            # mask
            mask = next((
                p for p in [
                    gt_dir / f"{stem}_mask.png",
                    gt_dir / f"{stem}.png"
                ] if p.exists()
            ), None)

            if mask:
                mask_name = f"{category}_{defect}_{stem}_mask.png"

                dst_mask = output_dir / "fail/masks" / mask_name
                transfer_file(mask, dst_mask, mode)
                stats["fail_masks"] += 1

                # annotated
                transfer_file(img, output_dir / "annotated/images" / name, mode)
                transfer_file(mask, output_dir / "annotated/masks" / mask_name, mode)
                stats["annotated"] += 1


def save_metadata(output_dir: Path, category: str, defect_types: List[str], stats: Dict, source: Path):
    meta = {
        "category": category,
        "defect_types": defect_types,
        "stats": dict(stats),
        "source": str(source),
    }

    path = output_dir / "meta" / f"{category}.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

# VLM TEMPLATE

def create_vlm_template(output_dir: Path, category: str, defect_types: List[str]):
    vlm_dir = output_dir / "vlm_annotated"
    vlm_dir.mkdir(parents=True, exist_ok=True)

    template = {
        "category": category,
        "defect_types": defect_types,
        "instruction": "Điền mô tả lỗi (tiếng Việt)",
        "sample": {
            "image": f"{category}_sample.png",
            "defect_type": "scratch",
            "description": "Vết xước nhỏ ở góc trên"
        }
    }

    out = vlm_dir / f"{category}_template.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)

    console.print(f"[green]VLM template → {out}[/green]")


# MAIN PROCESS

def process_category(root: Path, category: str, output_dir: Path, mode: str) -> Dict:
    stats = defaultdict(int)

    defect_types = get_defect_types(root, category)

    # pass
    process_pass_images(
        root / category / "train/good",
        output_dir / "pass/train",
        category, stats, mode, "pass_train"
    )

    process_pass_images(
        root / category / "test/good",
        output_dir / "pass/test",
        category, stats, mode, "pass_test"
    )

    # fail
    process_fail_images(root, category, defect_types, output_dir, stats, mode)

    # metadata
    save_metadata(output_dir, category, defect_types, stats, root / category)

    # vlm template
    create_vlm_template(output_dir, category, defect_types)

    return dict(stats)


# SUMMARY

def print_summary(all_stats: Dict):
    table = Table(title="Summary")

    table.add_column("Category")
    table.add_column("Train")
    table.add_column("Test")
    table.add_column("Fail")
    table.add_column("Mask")

    for cat, s in all_stats.items():
        table.add_row(
            cat,
            str(s.get("pass_train", 0)),
            str(s.get("pass_test", 0)),
            str(s.get("fail_images", 0)),
            str(s.get("fail_masks", 0)),
        )

    console.print(table)


# MAIN

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec_dir", required=True)
    parser.add_argument("--category", nargs="+")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--mode", default="copy", choices=["copy", "symlink"])

    args = parser.parse_args()

    root = Path(args.mvtec_dir)
    output_dir = Path(args.output_dir)

    if not root.exists():
        console.print("[red]Invalid path[/red]")
        return

    # categories
    if args.all:
        categories = MVTEC_CATEGORIES
    elif args.category:
        categories = args.category
    else:
        categories = ["metal_nut"]

    console.print(f"\n[bold]Processing:[/bold] {categories}\n")

    all_stats = {}

    for cat in categories:
        console.print(f"[cyan]→ {cat}[/cyan]")

        if not validate_category(root, cat):
            continue

        stats = process_category(root, cat, output_dir, args.mode)
        all_stats[cat] = stats

    print_summary(all_stats)

    console.print("\n[bold green]✔ Done[/bold green]")


if __name__ == "__main__":
    main()