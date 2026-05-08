"""
inference/batch_runner.py

Chạy pipeline trên toàn bộ thư mục ảnh — có thể xử lý hàng loạt.

Cách dùng:
    python inference/batch_runner.py --input data/fail/images/metal_nut/ --category metal_nut
    python inference/batch_runner.py --input data/ --pattern "**/*.png" --workers 1
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from configs.base_config import CATEGORY, DEVICE, RESULTS_DIR
from stage4_decision.engine import PipelineEngine
from stage4_decision.report import save_json_report, save_html_report, print_console_report

console = Console()


def find_images(input_dir: str, pattern: str = "*.png") -> List[Path]:
    """Tìm tất cả ảnh trong thư mục theo pattern."""
    base = Path(input_dir)
    images = sorted(base.glob(pattern))
    # Thêm các extension phổ biến nếu pattern mặc định
    if pattern == "*.png":
        for ext in ["*.jpg", "*.jpeg", "*.bmp"]:
            images += sorted(base.glob(ext))
        images = sorted(set(images))
    return images


def run_batch(
    images:      List[Path],
    category:    str,
    device:      str,
    output_dir:  str,
    stage1_only: bool = False,
    html:        bool = False,
    workers:     int  = 1,
) -> List[Dict]:
    """
    Chạy pipeline trên list ảnh.

    Args:
        workers: số thread song song (khuyến nghị = 1 với GPU để tránh OOM)
    """
    engine  = PipelineEngine(category=category, device=device, stage1_only=stage1_only)
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Processing {len(images)} images...", total=len(images))

        if workers == 1:
            # Sequential — an toàn với GPU
            for img_path in images:
                try:
                    result = engine.run(str(img_path))
                    save_json_report(result, output_dir)
                    if html:
                        save_html_report(result, output_dir)
                    results.append(result)
                except Exception as e:
                    console.print(f"[red]Error {img_path.name}: {e}[/red]")
                    results.append({"image_path": str(img_path), "error": str(e)})

                progress.advance(task)
        else:
            # Multi-thread — chỉ dùng CPU hoặc khi workers > 1
            def _run(img_path):
                try:
                    result = engine.run(str(img_path))
                    save_json_report(result, output_dir)
                    return result
                except Exception as e:
                    return {"image_path": str(img_path), "error": str(e)}

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_run, p): p for p in images}
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.advance(task)

    return results


def print_batch_summary(results: List[Dict]):
    """In bảng tổng kết sau khi batch xong."""
    total   = len(results)
    errors  = sum(1 for r in results if "error" in r)
    passed  = sum(1 for r in results if r.get("decision", {}).get("verdict") == "Pass")
    failed  = total - passed - errors

    # Priority distribution
    prio_count = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        if "decision" in r:
            p = r["decision"].get("priority", 0)
            prio_count[p] = prio_count.get(p, 0) + 1

    needs_review = sum(1 for r in results if r.get("decision", {}).get("needs_review"))

    avg_ms = np.mean([r.get("elapsed_ms", 0) for r in results if "elapsed_ms" in r])

    console.print("\n")

    table = Table(title="Batch Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold", width=22)
    table.add_column("Value", width=16)

    table.add_row("Total images",   str(total))
    table.add_row("[green]Pass[/]",  str(passed))
    table.add_row("[red]Fail[/]",    str(failed))
    table.add_row("Errors",          str(errors))
    table.add_row("Needs review",    str(needs_review))
    table.add_row("OK (priority 0)", str(prio_count[0]))
    table.add_row("Minor (1)",        str(prio_count[1]))
    table.add_row("Major (2)",        str(prio_count[2]))
    table.add_row("Critical (3)",     str(prio_count[3]))
    table.add_row("Avg time/image",   f"{avg_ms:.0f}ms")

    console.print(table)

    fail_rate = failed / max(total - errors, 1) * 100
    console.print(f"\nFail rate: [bold]{'🔴' if fail_rate > 20 else '🟢'} {fail_rate:.1f}%[/bold]")


def save_batch_report(results: List[Dict], output_dir: str, category: str):
    """Lưu tổng hợp toàn batch vào 1 JSON."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"batch_{category}_{timestamp}.json"

    summary = []
    for r in results:
        if "error" in r:
            summary.append({"image": r["image_path"], "error": r["error"]})
            continue
        dec = r.get("decision", {})
        summary.append({
            "image":        r.get("image_path"),
            "verdict":      dec.get("verdict"),
            "priority":     dec.get("priority"),
            "confidence":   dec.get("confidence"),
            "needs_review": dec.get("needs_review"),
            "elapsed_ms":   r.get("elapsed_ms"),
            "defect_type":  r.get("stage3", {}).get("defect_type"),
            "severity":     r.get("stage3", {}).get("severity"),
            "caption":      r.get("stage3", {}).get("caption"),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    console.print(f"[dim]Batch report → {out_path}[/dim]")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Defect Detection — Batch Inference")
    parser.add_argument("--input",       required=True,            help="Thư mục ảnh đầu vào")
    parser.add_argument("--category",    default=CATEGORY)
    parser.add_argument("--device",      default=DEVICE)
    parser.add_argument("--output",      default=RESULTS_DIR)
    parser.add_argument("--pattern",     default="*.png",           help="Glob pattern tìm ảnh")
    parser.add_argument("--stage1_only", action="store_true")
    parser.add_argument("--html",        action="store_true",        help="Xuất HTML cho từng ảnh")
    parser.add_argument("--workers",     type=int, default=1,        help="Số thread song song")
    parser.add_argument("--show_fails",  action="store_true",        help="In chi tiết ảnh Fail")
    args = parser.parse_args()

    images = find_images(args.input, args.pattern)
    if not images:
        console.print(f"[red]Không tìm thấy ảnh trong: {args.input}[/red]")
        return

    console.print(f"[bold]Batch Inference[/bold]")
    console.print(f"Found     : {len(images)} images")
    console.print(f"Category  : {args.category}")
    console.print(f"Output    : {args.output}\n")

    results = run_batch(
        images      = images,
        category    = args.category,
        device      = args.device,
        output_dir  = args.output,
        stage1_only = args.stage1_only,
        html        = args.html,
        workers     = args.workers,
    )

    print_batch_summary(results)
    save_batch_report(results, args.output, args.category)

    # In chi tiết ảnh Fail nếu yêu cầu
    if args.show_fails:
        fails = [r for r in results if r.get("decision", {}).get("verdict") == "Fail"]
        if fails:
            console.print(f"\n[bold red]Chi tiết {len(fails)} ảnh Fail:[/bold red]")
            for r in fails[:10]:   # Tối đa 10
                print_console_report(r)


if __name__ == "__main__":
    main()
