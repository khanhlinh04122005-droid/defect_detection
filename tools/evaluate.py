"""
tools/evaluate.py

Đánh giá toàn diện các stage:
    - Stage 1: AUROC, F1, Precision, Recall, Confusion Matrix
    - Stage 2: mIoU, mDice per defect type
    - Full pipeline: end-to-end accuracy

Cách dùng:
    python tools/evaluate.py --stage 1 --category metal_nut
    python tools/evaluate.py --stage 2 --category metal_nut
    python tools/evaluate.py --stage all --category metal_nut
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from configs.base_config import DEVICE, CATEGORY, MVTEC_DIR, EVAL_DIR
from configs.stage1_config import stage1_config

console = Console()


def eval_stage1(category: str, save: bool = True) -> dict:
    """
    Đánh giá Stage 1 trên test set MVTec.
    Cần: checkpoint đã train (outputs/checkpoints/stage1_<category>*)
    """
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        confusion_matrix, average_precision_score,
    )
    from stage1_anomaly.patchcore  import PatchCore
    from stage1_anomaly.extractor  import get_transform
    from stage1_anomaly.train      import TestDataset

    console.print(f"\n[bold cyan]Stage 1 Evaluation — {category}[/bold cyan]")

    model = PatchCore(category=category)
    model.load()

    transform  = get_transform(stage1_config["train"]["image_size"])
    test_dir   = f"{MVTEC_DIR}/{category}/test"
    test_ds    = TestDataset(test_dir, transform)

    console.print(f"Test images: {len(test_ds)}")

    gt_labels   = []
    pred_scores = []
    pred_labels = []

    for img_tensor, label in test_ds:
        tensor = img_tensor.unsqueeze(0).to(DEVICE)
        _, score = model._score_tensor(tensor)
        s = score.item()
        gt_labels.append(label)
        pred_scores.append(s)
        pred_labels.append(1 if s > model.bank.threshold else 0)

    gt     = np.array(gt_labels)
    scores = np.array(pred_scores)
    preds  = np.array(pred_labels)

    auroc = roc_auc_score(gt, scores)
    ap    = average_precision_score(gt, scores)
    f1    = f1_score(gt, preds, zero_division=0)
    prec  = precision_score(gt, preds, zero_division=0)
    rec   = recall_score(gt, preds, zero_division=0)
    cm    = confusion_matrix(gt, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    table = Table(title=f"Stage 1 — {category}", show_header=True, header_style="bold cyan")
    table.add_column("Metric",    style="bold", width=18)
    table.add_column("Score",     width=10)
    table.add_column("Target",    width=10)
    table.add_column("Status",    width=8)

    def row(name, val, target, fmt=".4f"):
        ok    = val >= target
        color = "green" if ok else "yellow"
        table.add_row(name, f"{val:{fmt}}", f"≥ {target}", f"[{color}]{'✔' if ok else '✗'}[/{color}]")

    row("AUROC",     auroc, 0.97)
    row("AP",        ap,    0.90)
    row("F1",        f1,    0.90)
    row("Precision", prec,  0.85)
    row("Recall",    rec,   0.95)
    console.print(table)

    console.print(f"\nConfusion Matrix:")
    console.print(f"  TN={tn}  FP={fp}")
    console.print(f"  FN={fn}  TP={tp}")

    results = dict(
        category=category, auroc=round(auroc,4), ap=round(ap,4),
        f1=round(f1,4), precision=round(prec,4), recall=round(rec,4),
        threshold=model.bank.threshold, n_test=len(test_ds),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )

    if save:
        out = Path(EVAL_DIR) / f"stage1_{category}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"\n[dim]Saved → {out}[/dim]")

    return results


def eval_stage2(category: str, save: bool = True) -> dict:
    """
    Đánh giá Stage 2 (segmentation) trên ảnh fail có ground-truth mask.
    """
    from stage1_anomaly.patchcore import PatchCore
    from stage2_seg.sam2_wrapper  import SAM2Wrapper
    from configs.base_config import FAIL_IMAGES_DIR, FAIL_MASKS_DIR

    console.print(f"\n[bold cyan]Stage 2 Evaluation — {category}[/bold cyan]")

    stage1 = PatchCore(category=category)
    stage1.load()
    stage2 = SAM2Wrapper(device=DEVICE, use_lora=True)

    img_dir  = Path(FAIL_IMAGES_DIR) / category
    mask_dir = Path(FAIL_MASKS_DIR)  / category

    samples = []
    for img_path in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            mask_path = mask_dir / (img_path.stem + ".png")
        if mask_path.exists():
            samples.append((img_path, mask_path))

    if not samples:
        console.print(f"[yellow]Không có mask trong {mask_dir}[/yellow]")
        return {}

    console.print(f"Samples with mask: {len(samples)}")

    ious, dices = [], []
    for img_path, mask_path in samples:
        img_np    = np.array(Image.open(img_path).convert("RGB"))
        gt_mask   = np.array(Image.open(mask_path).convert("L")) > 127

        result    = stage1.predict(str(img_path))
        pred_mask, _ = stage2.predict_mask(img_np, result["score_map"])

        inter = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        iou   = inter / (union + 1e-6)
        dice  = 2 * inter / (pred_mask.sum() + gt_mask.sum() + 1e-6)
        ious.append(iou); dices.append(dice)

    miou  = np.mean(ious)
    mdice = np.mean(dices)

    table = Table(title=f"Stage 2 — {category}", header_style="bold cyan")
    table.add_column("Metric", style="bold", width=14)
    table.add_column("Score",  width=10)
    table.add_column("Target", width=10)
    table.add_column("Status", width=8)

    for name, val, target in [("mIoU", miou, 0.80), ("mDice", mdice, 0.85)]:
        ok = val >= target
        c  = "green" if ok else "yellow"
        table.add_row(name, f"{val:.4f}", f"≥ {target}", f"[{c}]{'✔' if ok else '✗'}[/{c}]")
    console.print(table)

    results = dict(category=category, miou=round(miou,4), mdice=round(mdice,4), n_samples=len(samples))
    if save:
        out = Path(EVAL_DIR) / f"stage2_{category}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim]Saved → {out}[/dim]")

    return results


def eval_full(category: str, save: bool = True) -> dict:
    """End-to-end accuracy: Stage 1 + 2 + 3 + 4 cùng lúc."""
    from stage4_decision.engine import PipelineEngine
    from configs.base_config import FAIL_IMAGES_DIR, PASS_TEST_DIR

    console.print(f"\n[bold cyan]Full Pipeline Evaluation — {category}[/bold cyan]")

    engine = PipelineEngine(category=category, device=DEVICE)

    # Tập pass
    pass_dir = Path(PASS_TEST_DIR)
    pass_imgs = list(pass_dir.glob("*.png"))[:50]

    # Tập fail
    fail_dir  = Path(FAIL_IMAGES_DIR) / category
    fail_imgs = list(fail_dir.glob("*.png"))[:50]

    correct = 0
    total   = 0

    for imgs, gt in [(pass_imgs, "Pass"), (fail_imgs, "Fail")]:
        for img_path in imgs:
            try:
                r       = engine.run(str(img_path))
                verdict = r.get("decision", {}).get("verdict", "Pass")
                if verdict == gt:
                    correct += 1
                total += 1
            except Exception:
                total += 1

    acc = correct / max(total, 1)
    console.print(f"Accuracy: [bold]{acc:.2%}[/bold]  ({correct}/{total})")

    results = dict(category=category, accuracy=round(acc, 4), n_total=total)
    if save:
        out = Path(EVAL_DIR) / f"full_{category}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim]Saved → {out}[/dim]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate defect detection pipeline")
    parser.add_argument("--stage",    default="1", choices=["1", "2", "all"], help="Stage cần eval")
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--no_save",  action="store_true")
    args = parser.parse_args()

    save = not args.no_save
    cat  = args.category

    if args.stage == "1":
        eval_stage1(cat, save)
    elif args.stage == "2":
        eval_stage2(cat, save)
    elif args.stage == "all":
        r1 = eval_stage1(cat, save)
        r2 = eval_stage2(cat, save)
        eval_full(cat, save)


if __name__ == "__main__":
    main()
