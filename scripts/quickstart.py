#!/usr/bin/env python3
"""
scripts/quickstart_mvtec.py

Run full Stage 1 pipeline on MVTec:
prepare → train → evaluate → demo
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
from rich.console import Console
from rich.panel import Panel

console = Console()

# UTIL FUNCTIONS

def run_command(cmd: List[str], title: str) -> bool:
    """Run a shell command and print status."""
    console.print(f"\n[bold cyan]→ {title}[/bold cyan]")
    console.print(f"[dim]{' '.join(cmd)}[/dim]")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        console.print(f"[red]✖ Failed (code {result.returncode})[/red]")
        return False

    console.print("[green]✔ Done[/green]")
    return True


def validate_paths(mvtec_dir: Path):
    """Check if MVTec path exists."""
    if not mvtec_dir.exists():
        console.print(f"[red]MVTec path not found: {mvtec_dir}[/red]")
        sys.exit(1)


def get_data_paths(output_dir: Path):
    """Return structured dataset paths."""
    return {
        "pass_train": output_dir / "pass" / "train",
        "pass_test":  output_dir / "pass" / "test",
        "fail_img":   output_dir / "fail" / "images",
        "fail_mask":  output_dir / "fail" / "masks",
    }


# PIPELINE STEPS
def step_prepare(args) -> bool:
    return run_command([
        sys.executable, "scripts/prepare_mvtec.py",
        "--mvtec_dir", args.mvtec_dir,
        "--category", args.category,
        "--output_dir", args.output_dir,
        "--mode", "copy",
    ], "Prepare MVTec dataset")


def step_train(args, paths) -> bool:
    return run_command([
        sys.executable, "-m", "stage1_anomaly.train",
        "--data_dir", str(paths["pass_train"]),
        "--test_pass_dir", str(paths["pass_test"]),
        "--test_fail_dir", str(paths["fail_img"]),
        "--save_dir", "outputs/checkpoints/stage1",
    ], "Train PatchCore (Stage 1)")


def step_evaluate(args, paths) -> bool:
    return run_command([
        sys.executable, "tools/evaluate.py",
        "--test_pass", str(paths["pass_test"]),
        "--test_fail", str(paths["fail_img"]),
        "--mask_dir", str(paths["fail_mask"]),
        "--output", f"outputs/eval_{args.category}.json",
        "--stage1_only",
    ], "Evaluate model")


def step_demo(paths):
    fail_images = list(paths["fail_img"].glob("*.png"))

    if not fail_images:
        console.print("[yellow]No fail images found for demo[/yellow]")
        return

    sample = str(fail_images[0])

    run_command([
        sys.executable, "inference/run_pipeline.py",
        "--image", sample,
        "--output", "outputs/results/",
        "--stage1_only",
    ], f"Demo inference: {Path(sample).name}")


# MAIN


def main():
    parser = argparse.ArgumentParser(description="MVTec Quickstart Pipeline")

    parser.add_argument("--mvtec_dir", required=True)
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--output_dir", default="data/")
    parser.add_argument("--skip_prep", action="store_true")

    args = parser.parse_args()

    mvtec_dir = Path(args.mvtec_dir)
    output_dir = Path(args.output_dir)

    validate_paths(mvtec_dir)

    console.print(Panel(
        f"[bold]MVTec Quickstart[/bold]\n"
        f"Category : [cyan]{args.category}[/cyan]\n"
        f"MVTec dir: {mvtec_dir}\n"
        f"Output   : {output_dir}",
        title="Defect Detection"
    ))

    # Prepare
    if not args.skip_prep:
        if not step_prepare(args):
            return

    # Paths
    paths = get_data_paths(output_dir)

    # Train
    if not step_train(args, paths):
        return

    # Evaluate
    step_evaluate(args, paths)

    # Demo
    step_demo(paths)

    # Done
    console.print(Panel(
        "[bold green]✔ Pipeline completed[/bold green]\n\n"
        f"Checkpoint → outputs/checkpoints/stage1/\n"
        f"Eval       → outputs/eval_{args.category}.json\n"
        f"Results    → outputs/results/\n\n"
        "Next:\n"
        "• Check AUROC ≥ 97%\n"
        "• Move to Stage 2 (SAM2)",
        title="Done"
    ))


if __name__ == "__main__":
    main()