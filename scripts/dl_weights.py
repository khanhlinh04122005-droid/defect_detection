"""
scripts/dl_weights.py

Tự động download weights cần thiết về thư mục weights/.

Weights cần:
    - DINOv2 ViT-L/14  : tự download qua torch.hub (không cần script này)
    - SAM2-large        : download từ Meta GitHub releases
    - InternVL2-8B      : download từ HuggingFace (tuỳ chọn, rất nặng)

Cách dùng:
    python scripts/dl_weights.py --sam2
    python scripts/dl_weights.py --all
    python scripts/dl_weights.py --check
"""

import argparse
import hashlib
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, DownloadColumn, BarColumn, TransferSpeedColumn, TimeRemainingColumn

console = Console()

WEIGHTS_DIR = Path("weights")

SAM2_FILES = {
    "sam2_hiera_large.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "dest": WEIGHTS_DIR / "sam2" / "sam2_hiera_large.pt",
        "size_mb": 856,
    },
    "sam2_hiera_small.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "dest": WEIGHTS_DIR / "sam2" / "sam2_hiera_small.pt",
        "size_mb": 185,
    },
}

INTERNVL2_REPO = "OpenGVLab/InternVL2-8B"


def download_file(url: str, dest: Path, expected_mb: int = 0):
    """Download file với progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        console.print(f"[green]✔ Đã có:[/green] {dest.name} ({size_mb:.0f}MB)")
        return True

    console.print(f"[cyan]↓ Downloading:[/cyan] {dest.name} (~{expected_mb}MB)")
    console.print(f"  URL: {url}")

    try:
        with Progress(
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("", total=None)

            def _hook(block_num, block_size, total_size):
                if total_size > 0:
                    progress.update(task, total=total_size, completed=block_num * block_size)

            urllib.request.urlretrieve(url, dest, reporthook=_hook)

        size_mb = dest.stat().st_size / 1e6
        console.print(f"[green]✔ Done:[/green] {dest.name} ({size_mb:.0f}MB)")
        return True

    except Exception as e:
        console.print(f"[red]✖ Failed:[/red] {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_sam2(variant: str = "large"):
    """Download SAM2 checkpoint."""
    console.print("\n[bold]SAM2 Weights[/bold]")
    key = f"sam2_hiera_{variant}.pt"
    if key not in SAM2_FILES:
        console.print(f"[red]Unknown variant: {variant}. Options: large, small[/red]")
        return False
    info = SAM2_FILES[key]
    return download_file(info["url"], info["dest"], info["size_mb"])


def download_internvl2():
    """
    Download InternVL2-8B từ HuggingFace (rất nặng ~16GB).
    Cần: pip install huggingface_hub
    """
    console.print("\n[bold]InternVL2-8B (HuggingFace)[/bold]")
    console.print(f"[yellow]Cảnh báo: Model ~16GB, mất nhiều thời gian![/yellow]")

    dest = WEIGHTS_DIR / "internvl2"
    if dest.exists() and any(dest.iterdir()):
        console.print(f"[green]✔ Đã có:[/green] {dest}")
        return True

    try:
        from huggingface_hub import snapshot_download
        console.print(f"Downloading {INTERNVL2_REPO} → {dest}")
        snapshot_download(
            repo_id   = INTERNVL2_REPO,
            local_dir = str(dest),
            ignore_patterns = ["*.msgpack", "flax_model*", "tf_model*"],
        )
        console.print(f"[green]✔ InternVL2 downloaded → {dest}[/green]")
        return True
    except ImportError:
        console.print("[red]pip install huggingface_hub[/red]")
        return False
    except Exception as e:
        console.print(f"[red]✖ Failed: {e}[/red]")
        return False


def check_weights():
    """Kiểm tra trạng thái weights hiện có."""
    console.print("\n[bold]Weight Status[/bold]\n")

    checks = [
        ("SAM2-large",    WEIGHTS_DIR / "sam2" / "sam2_hiera_large.pt"),
        ("SAM2-small",    WEIGHTS_DIR / "sam2" / "sam2_hiera_small.pt"),
        ("InternVL2-8B",  WEIGHTS_DIR / "internvl2"),
    ]

    for name, path in checks:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1e6
                console.print(f"[green]✔[/green] {name:<20} {size:.0f}MB  ← {path}")
            else:
                n_files = len(list(path.rglob("*")))
                console.print(f"[green]✔[/green] {name:<20} ({n_files} files) ← {path}")
        else:
            console.print(f"[red]✗[/red] {name:<20} Chưa có")

    # DINOv2 — cache trong torch.hub
    import torch
    hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if hub_dir.exists():
        console.print(f"[green]✔[/green] {'DINOv2 (hub cache)':<20} ← {hub_dir}")
    else:
        console.print(f"[yellow]~[/yellow] {'DINOv2 (hub cache)':<20} Sẽ tự download khi chạy")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--sam2",      action="store_true", help="Download SAM2-large")
    parser.add_argument("--sam2_small",action="store_true", help="Download SAM2-small (nhẹ hơn)")
    parser.add_argument("--internvl2", action="store_true", help="Download InternVL2-8B (~16GB)")
    parser.add_argument("--all",       action="store_true", help="Download tất cả")
    parser.add_argument("--check",     action="store_true", help="Kiểm tra trạng thái weights")
    args = parser.parse_args()

    if args.check or not any([args.sam2, args.sam2_small, args.internvl2, args.all]):
        check_weights()
        return

    if args.all or args.sam2:
        download_sam2("large")

    if args.all or args.sam2_small:
        download_sam2("small")

    if args.all or args.internvl2:
        download_internvl2()

    console.print("\n[bold green]Done![/bold green]")
    check_weights()


if __name__ == "__main__":
    main()
