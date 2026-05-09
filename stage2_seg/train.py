"""
stage2_seg/train.py

Fine-tune SAM2 + LoRA trên ảnh fail có mask (TSfabrics).
Dùng anomaly score map từ Stage 1 làm prompt.

Cách dùng:
    python -m stage2_seg.train --category tsfabric_T1
    python -m stage2_seg.train --category tsfabric_T1 --eval
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from rich.console import Console

from configs.base_config import (
    DEVICE, NUM_WORKERS, SEED, CATEGORY,
    FAIL_IMAGES_DIR, FAIL_MASKS_DIR,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from configs.stage2_config import (
    TRAIN_BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, WARMUP_EPOCHS,
    WEIGHT_DECAY, MIN_LABELED_SAMPLES,
    DICE_LOSS_WEIGHT, BCE_LOSS_WEIGHT,
    AUG_FLIP, AUG_ROTATION, AUG_SCALE,
    LORA_WEIGHTS_PATH, TARGET_MIOU, TARGET_DICE,
)
from stage1_anomaly.patchcore import PatchCore
from stage2_seg.sam2_wrapper import SAM2Wrapper

console = Console()


class SegDataset(Dataset):
    """
    Dataset ảnh fail kèm ground-truth mask (TSfabrics).

    Cấu trúc thư mục (sau split_tsfabrics.py):
        data/fail/train/<category>/<filename>.jpeg   ← ảnh lỗi
        data/fail/masks/<category>/<filename>.png    ← binary mask, 0/255

    Fallback: nếu không có data/fail/train/, thử data/fail/images/ (layout cũ).
    """

    def __init__(self, category: str, augment: bool = True):
        self.category = category
        self.augment  = augment

        # TSfabrics layout: data/fail/train/<category>/
        img_dir = Path(FAIL_IMAGES_DIR).parent / "train" / category
        if not img_dir.exists():
            # Fallback layout cũ: data/fail/images/<category>/
            img_dir = Path(FAIL_IMAGES_DIR) / category

        mask_dir = Path(FAIL_MASKS_DIR) / category

        self.samples = []
        IMG_EXTS = ["*.jpeg", "*.jpg", "*.png", "*.bmp"]
        for ext in IMG_EXTS:
            for img_path in sorted(img_dir.glob(ext)):
                mask_path = mask_dir / (img_path.stem + ".png")
                if mask_path.exists():
                    self.samples.append((img_path, mask_path))

        console.print(f"[SegDataset] {len(self.samples)} ảnh fail có mask — category: {category}")
        console.print(f"             img_dir : {img_dir}")
        console.print(f"             mask_dir: {mask_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))   # (H, W, 3) uint8
        mask  = np.array(Image.open(mask_path).convert("L"))    # (H, W) uint8

        mask = (mask > 127).astype(np.float32)  # Binary [0, 1]

        if self.augment:
            image, mask = self._augment(image, mask)

        return image, mask, str(img_path)

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        h, w = image.shape[:2]

        # Flip ngang
        if AUG_FLIP and np.random.rand() > 0.5:
            image = image[:, ::-1].copy()
            mask  = mask[:, ::-1].copy()

        # Rotation
        if AUG_ROTATION > 0:
            angle  = np.random.uniform(-AUG_ROTATION, AUG_ROTATION)
            center = (w // 2, h // 2)
            M      = cv2.getRotationMatrix2D(center, angle, 1.0)
            image  = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
            mask   = cv2.warpAffine(mask,  M, (w, h), flags=cv2.INTER_NEAREST)

        # Scale
        if AUG_SCALE:
            scale = np.random.uniform(*AUG_SCALE)
            new_h, new_w = int(h * scale), int(w * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask  = cv2.resize(mask,  (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            # Crop / pad về kích thước gốc
            image = _center_crop_or_pad(image, h, w)
            mask  = _center_crop_or_pad(mask,  h, w)

        return image, mask


def _center_crop_or_pad(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    # Crop
    if h > target_h:
        start_h = (h - target_h) // 2
        arr = arr[start_h:start_h + target_h]
    if w > target_w:
        start_w = (w - target_w) // 2
        arr = arr[:, start_w:start_w + target_w]
    # Pad
    h, w = arr.shape[:2]
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h or pad_w:
        if arr.ndim == 3:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)))
        else:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)))
    return arr


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice Loss — pred và target đều trong [0,1]."""
    pred   = pred.flatten(1)
    target = target.flatten(1)
    inter  = (pred * target).sum(1)
    union  = pred.sum(1) + target.sum(1)
    return 1 - (2 * inter + eps) / (union + eps)


def seg_loss(pred_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Kết hợp BCE + Dice Loss.
    pred_logit: (B, 1, H, W) raw logit
    target:     (B, H, W)    binary float
    """
    target_4d = target.unsqueeze(1)
    bce = F.binary_cross_entropy_with_logits(pred_logit, target_4d, reduction="mean")
    dce = dice_loss(torch.sigmoid(pred_logit.squeeze(1)), target).mean()
    return BCE_LOSS_WEIGHT * bce + DICE_LOSS_WEIGHT * dce


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    return float(inter) / float(union + 1e-6)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    inter = (pred_mask & gt_mask).sum()
    return 2 * float(inter) / float(pred_mask.sum() + gt_mask.sum() + 1e-6)


def train(args):
    torch.manual_seed(SEED)
    category = args.category

    ds = SegDataset(category=category, augment=True)

    if len(ds) < MIN_LABELED_SAMPLES:
        console.print(
            f"[yellow]Chỉ có {len(ds)} ảnh có mask "
            f"(cần tối thiểu {MIN_LABELED_SAMPLES}).[/yellow]\n"
            f"Dùng SAM2 zero-shot, bỏ qua fine-tune."
        )
        return

    dl = DataLoader(
        ds,
        batch_size  = TRAIN_BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,   # Windows: phải là 0
        collate_fn  = _collate,
    )

    # Load Stage 1 trên CPU — chỉ cần score map, tránh chiếm VRAM của SAM2
    console.print("[Stage2] Loading Stage 1 PatchCore (CPU)...")
    stage1 = PatchCore(category=category, device="cpu")
    stage1.load()

    # Khởi tạo SAM2 + LoRA
    console.print("[Stage2] Loading SAM2 + LoRA...")
    model = SAM2Wrapper(device=DEVICE, use_lora=True)
    model.train()

    # Chỉ train LoRA params
    from stage2_seg.lora_adapter import get_lora_params
    optimizer = torch.optim.AdamW(
        get_lora_params(model.image_encoder),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )
    # bfloat16 không cần loss scaling (khác float16) — disable scaler
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS
    )

    # SAM2 official: dùng bfloat16 (Ampere GPU) + tf32
    use_amp = DEVICE == "cuda"
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    console.print(f"\n[bold]Stage 2 — Fine-tune SAM2 LoRA[/bold]")
    console.print(f"Category  : {category}")
    console.print(f"Epochs    : {NUM_EPOCHS}  |  Batch: {TRAIN_BATCH_SIZE}  |  LR: {LEARNING_RATE}")
    console.print(f"AMP       : {'bfloat16' if use_amp else 'disabled'}")
    console.print(f"Loss      : BCE×{BCE_LOSS_WEIGHT} + Dice×{DICE_LOSS_WEIGHT}\n")

    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0

        for images, masks, paths in dl:
            optimizer.zero_grad()
            step_loss = 0.0

            for img_np, gt_mask_np, path in zip(images, masks, paths):
                result    = stage1.predict(path)
                score_map = result["score_map"]

                h, w   = img_np.shape[:2]
                prompt = model.score_map_to_prompt(score_map, (h, w))
                if not prompt:
                    continue

                sam_model = model.predictor.model

                # Toàn bộ forward trong bfloat16 autocast — SAM2 official pattern
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    features, img_pe = model.encode_for_training(img_np)

                    with torch.no_grad():
                        sparse_emb, dense_emb = sam_model.sam_prompt_encoder(
                            points=(
                                torch.from_numpy(prompt["point_coords"]).float().unsqueeze(0).to(DEVICE),
                                torch.from_numpy(prompt["point_labels"]).float().unsqueeze(0).to(DEVICE),
                            ) if "point_coords" in prompt else None,
                            boxes=torch.from_numpy(prompt["box"]).float().unsqueeze(0).to(DEVICE)
                                  if "box" in prompt else None,
                            masks=None,
                        )

                    low_res_logits, _, _, _ = sam_model.sam_mask_decoder(
                        image_embeddings=features["image_embed"],
                        image_pe=img_pe,
                        sparse_prompt_embeddings=sparse_emb,
                        dense_prompt_embeddings=dense_emb,
                        multimask_output=False,
                        repeat_image=False,
                        high_res_features=features["high_res_feats"],
                    )

                    logit_up  = F.interpolate(
                        low_res_logits.float(),   # cast float32 trước khi tính loss
                        size=(h, w), mode="bilinear", align_corners=False,
                    )
                    gt_tensor = torch.from_numpy(gt_mask_np).unsqueeze(0).to(DEVICE)
                    loss      = seg_loss(logit_up, gt_tensor)

                loss.backward()   # backward ngay sau mỗi ảnh (batch_size=1)
                step_loss += loss.item()

            if step_loss > 0:
                torch.nn.utils.clip_grad_norm_(model.image_encoder.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += step_loss

        if epoch > WARMUP_EPOCHS:
            scheduler.step()

        console.print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"loss={epoch_loss / max(len(dl), 1):.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    console.print(f"\nTổng thời gian: {time.time() - t0:.1f}s")

    # Lưu LoRA weights
    model.save_lora()
    console.print(f"[bold green]LoRA weights saved → {LORA_WEIGHTS_PATH}[/bold green]")

    if args.eval:
        _evaluate(model, stage1, category)


def _collate(batch):
    """Collate function cho SegDataset (giữ list vì kích thước ảnh có thể khác nhau)."""
    images = [b[0] for b in batch]
    masks  = [b[1] for b in batch]
    paths  = [b[2] for b in batch]
    return images, masks, paths


def _evaluate(model: SAM2Wrapper, stage1: PatchCore, category: str):
    console.print("\n[bold]Evaluation — Stage 2[/bold]")

    ds = SegDataset(category=category, augment=False)
    if not ds.samples:
        console.print("[red]Không có ảnh để evaluate.[/red]")
        return

    model.eval()
    ious, dices = [], []
    results = []

    for img_np, gt_mask, img_path in ds:
        # Score map từ Stage 1
        result    = stage1.predict(img_path)
        score_map = result["score_map"]

        # Predict mask Stage 2
        pred_mask, sam_score = model.predict_mask(img_np, score_map)
        gt_bool = gt_mask.astype(bool)

        iou   = compute_iou(pred_mask, gt_bool)
        dice  = compute_dice(pred_mask, gt_bool)
        ious.append(iou)
        dices.append(dice)

        results.append({
            "image": img_path,
            "iou":   round(iou, 4),
            "dice":  round(dice, 4),
            "sam_score": round(sam_score, 4),
        })

    mean_iou  = np.mean(ious)
    mean_dice = np.mean(dices)

    console.print(f"mIoU : [bold]{mean_iou:.4f}[/bold]  (target ≥ {TARGET_MIOU})")
    console.print(f"mDice: [bold]{mean_dice:.4f}[/bold]  (target ≥ {TARGET_DICE})")

    if mean_iou >= TARGET_MIOU:
        console.print("[bold green]✓ mIoU target đạt![/bold green]")
    else:
        console.print("[yellow]✗ mIoU chưa đạt target[/yellow]")

    # Lưu kết quả
    out = Path(RESULTS_DIR) / f"stage2_{category}_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mean_iou": round(mean_iou, 4), "mean_dice": round(mean_dice, 4), "per_image": results}, f, indent=2)
    console.print(f"Kết quả lưu → {out}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2 — SAM2 + LoRA fine-tune (TSfabrics)")
    parser.add_argument("--category", default=CATEGORY,
                        help="Loại vải: tsfabric_T1 / T2 / T3 (mặc định: tsfabric_T1)")
    parser.add_argument("--eval",     action="store_true", help="Đánh giá sau khi train")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
