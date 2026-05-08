"""
inference/run_pipeline.py

CLI inference cho 1 ảnh — chạy full 4-stage pipeline.

Cách dùng:
    python inference/run_pipeline.py --image data/fail/images/metal_nut/001.png
    python inference/run_pipeline.py --image path/to/img.jpg --stage1_only
    python inference/run_pipeline.py --image path/to/img.jpg --html
"""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from configs.base_config import CATEGORY, DEVICE, RESULTS_DIR
from stage4_decision.engine import PipelineEngine
from stage4_decision.report import (
    print_console_report,
    save_json_report,
    save_html_report,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Defect Detection — Single Image Inference")
    parser.add_argument("--image",       required=True,           help="Đường dẫn ảnh cần kiểm tra")
    parser.add_argument("--category",    default=CATEGORY,        help="MVTec category")
    parser.add_argument("--device",      default=DEVICE,          help="cuda | cpu")
    parser.add_argument("--output",      default=RESULTS_DIR,     help="Thư mục lưu kết quả")
    parser.add_argument("--stage1_only", action="store_true",      help="Chỉ chạy Stage 1")
    parser.add_argument("--stage12_only",action="store_true",      help="Chỉ chạy Stage 1+2")
    parser.add_argument("--html",        action="store_true",      help="Xuất thêm HTML report")
    parser.add_argument("--no_json",     action="store_true",      help="Không lưu JSON")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        console.print(f"[red]Không tìm thấy ảnh: {img_path}[/red]")
        sys.exit(1)

    console.print(f"[bold]Defect Detection Pipeline[/bold]")
    console.print(f"Image   : {img_path}")
    console.print(f"Category: {args.category}")
    console.print(f"Device  : {args.device}\n")

    engine = PipelineEngine(
        category     = args.category,
        device       = args.device,
        stage1_only  = args.stage1_only,
        stage12_only = args.stage12_only,
    )

    result = engine.run(str(img_path))

    # Console report
    print_console_report(result)

    # JSON
    if not args.no_json:
        json_path = save_json_report(result, args.output)
        console.print(f"\n[dim]JSON → {json_path}[/dim]")

    # HTML
    if args.html:
        html_path = save_html_report(result, args.output)
        console.print(f"[dim]HTML → {html_path}[/dim]")

    # Exit code: 1 nếu Fail
    verdict = result.get("decision", {}).get("verdict", "Pass")
    sys.exit(0 if verdict == "Pass" else 1)


if __name__ == "__main__":
    main()
