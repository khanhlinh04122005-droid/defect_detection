"""
scripts/tune_threshold.py

Tune threshold Stage 1 mà không cần rebuild memory bank.
Load checkpoint sẵn → thử nhiều percentile → tìm percentile tối ưu F1.

Dùng:
    python scripts/tune_threshold.py --category tsfabric_T1
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import numpy as np
from pathlib import Path

from configs.base_config import DEVICE, DATA_DIR, CHECKPOINT_DIR, EVAL_DIR
from configs.stage1_config import stage1_config
from stage1_anomaly.extractor import get_transform
from stage1_anomaly.patchcore import PatchCore
from stage1_anomaly.train import ImageFolderDataset, EvalDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score


def tune(args):
    category  = args.category
    transform = get_transform(stage1_config["train"]["image_size"])

    print(f"Loading checkpoint: {CHECKPOINT_DIR}/stage1_{category}")
    model = PatchCore(category=category)
    model.load()

    # Score toàn bộ val pass set
    val_dir = Path(DATA_DIR) / "pass" / "test" / category
    val_ds  = ImageFolderDataset(val_dir, transform, label=0)
    val_dl  = DataLoader(val_ds, batch_size=1, num_workers=0)
    print(f"Val pass images: {len(val_ds)}")

    val_scores = []
    for batch in val_dl:
        img = batch[0].to(DEVICE)
        _, score = model._score_tensor(img)
        val_scores.append(score.item())
    val_scores = np.array(val_scores)

    # Score eval set (pass + fail)
    eval_ds = EvalDataset(category, transform)
    gt_labels, pred_scores = [], []
    print(f"Eval images: {len(eval_ds)}")
    for img_tensor, label in eval_ds:
        tensor = img_tensor.unsqueeze(0).to(DEVICE)
        _, score = model._score_tensor(tensor)
        gt_labels.append(label)
        pred_scores.append(score.item())

    gt     = np.array(gt_labels)
    scores = np.array(pred_scores)
    auroc  = roc_auc_score(gt, scores)
    print(f"AUROC: {auroc:.4f}\n")

    # Thử các percentile
    print(f"{'Percentile':>11} {'Threshold':>12} {'F1':>8} {'Precision':>10} {'Recall':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 75)

    best_f1, best_pct = 0, 99.0
    for pct in [70, 75, 80, 85, 88, 90, 92, 95, 97, 99]:
        thr   = float(np.percentile(val_scores, pct))
        preds = (scores > thr).astype(int)
        f1    = f1_score(gt, preds, zero_division=0)
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        prec  = precision_score(gt, preds, zero_division=0)
        rec   = recall_score(gt, preds, zero_division=0)
        cm    = confusion_matrix(gt, preds, labels=[0, 1])
        tp    = cm[1, 1] if cm.shape == (2, 2) else 0
        fp    = cm[0, 1] if cm.shape == (2, 2) else 0
        fn    = cm[1, 0] if cm.shape == (2, 2) else 0
        print(f"{pct:>11} {thr:>12.2f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f} {tp:>5} {fp:>5} {fn:>5}")
        if f1 > best_f1:
            best_f1, best_pct = f1, pct

    print(f"\nBest percentile: {best_pct}  (F1={best_f1:.4f})")

    if args.apply:
        best_thr = float(np.percentile(val_scores, best_pct))
        model.bank.threshold = best_thr
        model.save()
        print(f"Threshold updated → {best_thr:.4f} (percentile={best_pct})")
        print(f"Saved → {CHECKPOINT_DIR}/stage1_{category}")
        print(f"\nUpdate configs/stage1_config.py:")
        print(f'  "tsfabric_T1": {{"feature_layers": [12, 18], "k_nearest": 9, "threshold_percentile": {best_pct}.0}}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="tsfabric_T1")
    parser.add_argument("--apply", action="store_true", help="Lưu threshold tốt nhất vào checkpoint")
    tune(parser.parse_args())
