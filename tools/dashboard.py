"""
tools/dashboard.py

Dashboard thống kê kết quả batch inference.
Đọc các file JSON trong outputs/results/ và outputs/eval/
rồi tổng hợp thành bảng đẹp với rich.

Cách dùng:
    python tools/dashboard.py
    python tools/dashboard.py --category metal_nut
    python tools/dashboard.py --watch    # Tự refresh mỗi 10s
"""

import argparse
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.columns import Columns
from rich.text import Text

from configs.base_config import RESULTS_DIR, EVAL_DIR, CATEGORY

console = Console()


def load_batch_results(results_dir: str = RESULTS_DIR, category: str = None) -> List[Dict]:
    """Đọc tất cả JSON kết quả trong thư mục results."""
    path    = Path(results_dir)
    results = []
    pattern = f"*{category}*" if category else "*.json"
    for f in sorted(path.glob(pattern)):
        if f.name.startswith("batch_"):
            try:
                with open(f, encoding="utf-8") as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    results.extend(data)
            except Exception:
                pass
    return results


def load_eval_results(eval_dir: str = EVAL_DIR, category: str = None) -> Dict:
    """Đọc JSON eval của từng stage."""
    path  = Path(eval_dir)
    evals = {}
    for stage in ["1", "2", "3"]:
        pattern = f"stage{stage}_{category}.json" if category else f"stage{stage}_*.json"
        files   = list(path.glob(pattern))
        if files:
            with open(files[-1], encoding="utf-8") as f:
                evals[f"stage{stage}"] = json.load(f)
    return evals


def render_eval_table(evals: Dict) -> Table:
    """Bảng kết quả đánh giá các stage."""
    table = Table(title="Model Performance", header_style="bold cyan", show_lines=True)
    table.add_column("Stage",  style="bold", width=18)
    table.add_column("Metric", width=14)
    table.add_column("Score",  width=10)
    table.add_column("Target", width=10)
    table.add_column("",       width=4)

    metrics = {
        "stage1": [("AUROC", "auroc", 0.97), ("F1", "f1", 0.90), ("Recall", "recall", 0.95)],
        "stage2": [("mIoU",  "miou",  0.80), ("mDice","mdice",0.85)],
    }

    for stage_key, stage_metrics in metrics.items():
        data = evals.get(stage_key, {})
        for i, (name, key, target) in enumerate(stage_metrics):
            val  = data.get(key)
            if val is None:
                table.add_row(stage_key if i == 0 else "", name, "[dim]N/A[/dim]", f"≥{target}", "")
                continue
            ok   = val >= target
            color = "green" if ok else "red"
            table.add_row(
                stage_key if i == 0 else "",
                name,
                f"[{color}]{val:.4f}[/{color}]",
                f"≥ {target}",
                f"[{color}]{'✔' if ok else '✗'}[/{color}]",
            )

    return table


def render_verdict_summary(results: List[Dict]) -> Table:
    """Bảng tóm tắt Pass/Fail từ batch results."""
    if not results:
        table = Table(title="Batch Results")
        table.add_column("Info")
        table.add_row("[dim]Chưa có kết quả batch.[/dim]")
        return table

    total  = len(results)
    passed = sum(1 for r in results if r.get("verdict") == "Pass")
    failed = total - passed

    priority_count = Counter(r.get("priority", 0) for r in results)
    defect_counter = Counter(r.get("defect_type") for r in results if r.get("defect_type") and r.get("defect_type") != "unknown")
    review_count   = sum(1 for r in results if r.get("needs_review"))
    errors         = sum(1 for r in results if "error" in r)

    avg_ms    = np.mean([r.get("elapsed_ms", 0) for r in results if r.get("elapsed_ms")]) if results else 0
    fail_rate = failed / max(total - errors, 1) * 100

    table = Table(title=f"Batch Summary ({total} images)", header_style="bold cyan")
    table.add_column("Metric",   style="bold", width=22)
    table.add_column("Value",    width=16)

    c = "green" if fail_rate < 10 else "yellow" if fail_rate < 30 else "red"
    table.add_row("[green]Pass[/]",           str(passed))
    table.add_row("[red]Fail[/]",             str(failed))
    table.add_row(f"Fail rate",               f"[{c}]{fail_rate:.1f}%[/{c}]")
    table.add_row("Needs review",             str(review_count))
    table.add_row("Errors",                   str(errors))
    table.add_row("Priority: OK (0)",         str(priority_count.get(0, 0)))
    table.add_row("Priority: Minor (1)",       str(priority_count.get(1, 0)))
    table.add_row("Priority: Major (2)",       str(priority_count.get(2, 0)))
    table.add_row("Priority: Critical (3)",    str(priority_count.get(3, 0)))
    table.add_row("Avg time/image",           f"{avg_ms:.0f}ms")

    return table


def render_defect_types(results: List[Dict]) -> Table:
    """Bảng phân bố loại lỗi."""
    table = Table(title="Defect Types (Fail images)", header_style="bold cyan")
    table.add_column("Defect Type", style="bold", width=20)
    table.add_column("Count",       width=8, justify="right")
    table.add_column("Severity",    width=30)

    if not results:
        table.add_row("[dim]Chưa có dữ liệu[/dim]", "", "")
        return table

    type_sev: Dict[str, Counter] = defaultdict(Counter)
    for r in results:
        if r.get("verdict") != "Fail":
            continue
        dt  = r.get("defect_type", "unknown") or "unknown"
        sev = r.get("severity", "unknown") or "unknown"
        type_sev[dt][sev] += 1

    for dt, sev_counts in sorted(type_sev.items(), key=lambda x: -sum(x[1].values())):
        total = sum(sev_counts.values())
        sev_str = "  ".join(f"{s}:{n}" for s, n in sev_counts.most_common(3))
        table.add_row(dt, str(total), f"[dim]{sev_str}[/dim]")

    if not type_sev:
        table.add_row("[dim]Không có ảnh Fail[/dim]", "", "")

    return table


def render_recent_fails(results: List[Dict], n: int = 10) -> Table:
    """Danh sách ảnh Fail gần nhất cần review."""
    fails = [r for r in results if r.get("verdict") == "Fail"]
    fails = sorted(fails, key=lambda r: r.get("priority", 0), reverse=True)[:n]

    table = Table(title=f"Top {n} Fails (by priority)", header_style="bold red")
    table.add_column("Image",        width=30)
    table.add_column("Priority",     width=10)
    table.add_column("Defect",       width=15)
    table.add_column("Severity",     width=10)
    table.add_column("Conf",         width=8)
    table.add_column("Review",       width=8)

    prio_colors = {0:"green", 1:"yellow", 2:"orange1", 3:"red"}
    prio_labels = {0:"OK", 1:"Minor", 2:"Major", 3:"Critical"}

    for r in fails:
        p   = r.get("priority", 0)
        c   = prio_colors.get(p, "white")
        rv  = "⚠" if r.get("needs_review") else ""
        table.add_row(
            Path(r.get("image", r.get("image_path", "?"))).name[:28],
            f"[{c}]{prio_labels.get(p,'?')}[/{c}]",
            r.get("defect_type", "?"),
            r.get("severity", "?"),
            f"{r.get('confidence', 0):.2f}",
            f"[yellow]{rv}[/yellow]",
        )

    if not fails:
        table.add_row("[green]Không có ảnh Fail[/green]", "", "", "", "", "")

    return table


def show_dashboard(category: Optional[str] = None, watch: bool = False):
    """Hiển thị đầy đủ dashboard."""
    def _render():
        results = load_batch_results(category=category)
        evals   = load_eval_results(category=category)
        ts      = time.strftime("%Y-%m-%d %H:%M:%S")

        console.clear()
        console.print(Panel(
            f"[bold]Defect Detection Dashboard[/bold]  |  "
            f"Category: [cyan]{category or 'all'}[/cyan]  |  "
            f"[dim]{ts}[/dim]",
            style="bold blue",
        ))

        # Row 1: Eval + Summary
        console.print(Columns([
            render_eval_table(evals),
            render_verdict_summary(results),
        ], equal=False, expand=False))

        # Row 2: Defect types + Recent fails
        console.print(Columns([
            render_defect_types(results),
            render_recent_fails(results),
        ], equal=False, expand=False))

        if watch:
            console.print(f"\n[dim]Auto-refresh mỗi 10s | Ctrl+C để thoát[/dim]")

    if watch:
        try:
            while True:
                _render()
                time.sleep(10)
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped.[/yellow]")
    else:
        _render()


def main():
    parser = argparse.ArgumentParser(description="Defect Detection Dashboard")
    parser.add_argument("--category", default=None)
    parser.add_argument("--watch",    action="store_true", help="Auto refresh mỗi 10s")
    args = parser.parse_args()
    show_dashboard(args.category, args.watch)


if __name__ == "__main__":
    main()
