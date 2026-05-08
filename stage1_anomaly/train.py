"""
stage1_anomaly/train.py

Train Stage 1: build memory bank từ ảnh pass của TSfabrics,
tính threshold, evaluate trên test set, lưu kết quả.

Cấu trúc thư mục (sau khi chạy split_tsfabrics.py):
  data/pass/train/<category>/   → Ảnh pass dùng build Memory Bank
  data/pass/test/<category>/    → Ảnh pass dùng tính threshold
  data/pass/eval/<category>/    → Ảnh pass dùng đánh giá AUROC
  data/fail/test/<category>/    → Ảnh fail dùng đánh giá AUROC

Cách chạy:
    python -m stage1_anomaly.train
    python -m stage1_anomaly.train --category tsfabric_T1 --eval
    python -m stage1_anomaly.train --category tsfabric_T2 --eval
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from rich.console import Console
from rich.table   import Table

from configs.base_config  import (
    DEVICE, SEED, NUM_WORKERS, CATEGORY,
    EVAL_DIR, CHECKPOINT_DIR, DATA_DIR,
)
from configs.stage1_config import stage1_config, category_tuning
from stage1_anomaly.extractor  import get_transform
from stage1_anomaly.patchcore  import PatchCore

console = Console()


# ── Datasets ─────────────────────────────────────────────────────────

class ImageFolderDataset(Dataset):
    """
    Đọc tất cả ảnh (.jpeg/.jpg/.png) từ một thư mục.
    Hỗ trợ cả flat và nested (glob **).
    """
    EXTS = ["*.jpeg", "*.jpg", "*.png", "*.bmp"]

    def __init__(self, folder: str | Path, transform, label: int = 0):
        self.transform = transform
        self.label     = label
        folder = Path(folder)
        self.paths = []
        for ext in self.EXTS:
            self.paths += sorted(folder.glob(f"**/{ext}"))
        self.paths = sorted(set(self.paths))   # loại bỏ trùng lặp

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.label


class EvalDataset(Dataset):
    """
    Gộp ảnh pass (label=0) và ảnh fail (label=1) để đánh giá AUROC.

    data/pass/eval/<category>/  →  label 0
    data/fail/test/<category>/  →  label 1
    """
    EXTS = ["*.jpeg", "*.jpg", "*.png", "*.bmp"]

    def __init__(self, category: str, transform):
        self.transform = transform
        self.samples   = []

        data = Path(DATA_DIR)
        for folder, lbl in [
            (data / "pass" / "eval" / category, 0),
            (data / "fail" / "test" / category, 1),
        ]:
            if not folder.exists():
                console.print(f"  [yellow]Không tìm thấy: {folder}[/yellow]")
                continue
            for ext in self.EXTS:
                for p in sorted(folder.glob(f"**/{ext}")):
                    self.samples.append((p, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Train ─────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(SEED)
    category  = args.category
    img_size  = stage1_config["train"]["image_size"]
    batch_size = stage1_config["train"]["batch_size"]

    console.print(f"\n[bold cyan]Stage 1 — PatchCore + DINOv2[/bold cyan]")
    console.print(f"Category  : [bold]{category}[/bold]")
    console.print(f"Image size: {img_size}px  |  Batch: {batch_size}")
    console.print(f"Device    : {DEVICE}\n")

    transform = get_transform(img_size)

    # ── 1. Build Memory Bank ──────────────────────────────────────────
    train_dir = Path(DATA_DIR) / "pass" / "train" / category
    if not train_dir.exists():
        console.print(
            f"[red]Không tìm thấy: {train_dir}\n"
            f"Hãy chạy trước: python scripts/split_tsfabrics.py[/red]"
        )
        return

    train_ds = ImageFolderDataset(train_dir, transform, label=0)
    # Windows + CPU: dùng num_workers=0 để tránh lỗi paging file
    _workers = 0 if DEVICE == "cpu" else NUM_WORKERS
    train_dl = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = _workers,
        pin_memory  = False,   # pin_memory chỉ có tác dụng khi có GPU
    )
    console.print(f"Train images (pass): [bold]{len(train_ds)}[/bold]")

    model = PatchCore(category=category)
    t0    = time.time()
    model.fit(train_dl)
    console.print(f"Memory bank built in {time.time()-t0:.1f}s")

    # ── 2. Tính Threshold ─────────────────────────────────────────────
    val_dir = Path(DATA_DIR) / "pass" / "test" / category
    if not val_dir.exists():
        console.print(f"[yellow]Không tìm thấy val dir: {val_dir} — dùng threshold mặc định.[/yellow]")
        threshold = None
    else:
        val_ds = ImageFolderDataset(val_dir, transform, label=0)
        val_dl = DataLoader(val_ds, batch_size=1, num_workers=0)
        console.print(f"Val images (pass/test): [bold]{len(val_ds)}[/bold]")
        threshold = model.fit_threshold(val_dl)
        console.print(f"Threshold: [bold]{threshold:.4f}[/bold]")

    model.save()
    console.print(f"[green]Model saved → {CHECKPOINT_DIR}/{category}[/green]")

    if args.eval:
        _evaluate(model, category, transform)


# ── Evaluate ──────────────────────────────────────────────────────────

def _evaluate(model: PatchCore, category: str, transform):
    console.print(f"\n[bold]Evaluation — {category}[/bold]")

    test_ds = EvalDataset(category, transform)
    n_pass  = sum(1 for _, l in test_ds.samples if l == 0)
    n_fail  = sum(1 for _, l in test_ds.samples if l == 1)
    console.print(f"Test: [green]{n_pass} pass[/green] + [red]{n_fail} fail[/red] = {len(test_ds)} ảnh")

    if n_fail == 0:
        console.print("[red]Không có ảnh fail để tính AUROC. Hãy chạy split_tsfabrics.py trước.[/red]")
        return

    gt_labels   = []
    pred_scores = []

    for img_tensor, label in test_ds:
        tensor = img_tensor.unsqueeze(0).to(DEVICE)
        _, score = model._score_tensor(tensor)
        gt_labels.append(label)
        pred_scores.append(score.item())

    gt     = np.array(gt_labels)
    scores = np.array(pred_scores)
    preds  = (scores > model.bank.threshold).astype(int)

    auroc  = roc_auc_score(gt, scores)
    f1     = f1_score(gt, preds, zero_division=0)
    prec   = precision_score(gt, preds, zero_division=0)
    recall = recall_score(gt, preds, zero_division=0)

    # Tuning target của category này
    tuning = category_tuning.get(category, {})

    table = Table(title=f"Stage 1 Results — {category}")
    table.add_column("Metric",    style="cyan")
    table.add_column("Score",     style="bold")
    table.add_column("Target")
    table.add_row("AUROC",     f"{auroc:.4f}",  "≥ 0.97")
    table.add_row("F1",        f"{f1:.4f}",     "≥ 0.90")
    table.add_row("Precision", f"{prec:.4f}",   "≥ 0.85")
    table.add_row("Recall",    f"{recall:.4f}", "≥ 0.95")
    table.add_row("k_nearest", str(tuning.get("k_nearest", "-")), "-")
    table.add_row("Percentile",str(tuning.get("threshold_percentile", "-")), "-")
    console.print(table)

    results = {
        "category"  : category,
        "dataset"   : "tsfabrics",
        "auroc"     : round(auroc,  4),
        "f1"        : round(f1,     4),
        "precision" : round(prec,   4),
        "recall"    : round(recall, 4),
        "threshold" : round(model.bank.threshold, 6),
        "n_pass"    : n_pass,
        "n_fail"    : n_fail,
    }
    out = Path(EVAL_DIR) / f"stage1_{category}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\nResults saved → [bold]{out}[/bold]")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1 — PatchCore train (TSfabrics)")
    parser.add_argument(
        "--category", default=CATEGORY,
        help="Loại vải TSfabrics: tsfabric_T1 / tsfabric_T2 / tsfabric_T3 (mặc định: tsfabric_T1)"
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate sau khi build Memory Bank")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()