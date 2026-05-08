"""
stage3_vlm/lora_train.py

Fine-tune InternVL2-8B + LoRA để sinh caption mô tả lỗi sản phẩm.

Cách dùng:
    python -m stage3_vlm.lora_train --category metal_nut
    python -m stage3_vlm.lora_train --category metal_nut --eval
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from rich.console import Console

from configs.base_config import DEVICE, SEED, CATEGORY, RESULTS_DIR
from configs.stage3_config import (
    TRAIN_BATCH_SIZE, GRAD_ACCUMULATION, LEARNING_RATE,
    NUM_EPOCHS, WARMUP_RATIO, WEIGHT_DECAY, MAX_GRAD_NORM,
    MIN_CAPTIONED_SAMPLES, LORA_WEIGHTS_PATH,
)
from stage3_vlm.model       import InternVL2Wrapper
from stage3_vlm.dataset_vlm import VLMDataset, collate_vlm

console = Console()


def train(args):
    torch.manual_seed(SEED)

    # Dataset
    ds = VLMDataset(augment=True)

    if len(ds) < MIN_CAPTIONED_SAMPLES:
        console.print(
            f"[yellow]Chỉ có {len(ds)} ảnh có caption "
            f"(cần ≥ {MIN_CAPTIONED_SAMPLES}).\n"
            f"Dùng InternVL2 zero-shot, bỏ qua fine-tune.[/yellow]"
        )
        return

    model = InternVL2Wrapper(device=DEVICE, use_lora=True)
    model.train_mode()

    # Tokenize dataset với tokenizer từ model
    ds_tok = VLMDataset(tokenizer=model.tokenizer, augment=True)
    dl = DataLoader(
        ds_tok,
        batch_size  = TRAIN_BATCH_SIZE,
        shuffle     = True,
        num_workers = 2,
        collate_fn  = collate_vlm,
    )

    # Optimizer — chỉ train LoRA params
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr           = LEARNING_RATE,
        weight_decay = WEIGHT_DECAY,
    )

    total_steps   = NUM_EPOCHS * len(dl) // GRAD_ACCUMULATION
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = _get_cosine_schedule(optimizer, warmup_steps, total_steps)

    console.print(f"\n[bold]Stage 3 — Fine-tune InternVL2 LoRA[/bold]")
    console.print(f"Samples   : {len(ds_tok)}")
    console.print(f"Epochs    : {NUM_EPOCHS}  |  Batch: {TRAIN_BATCH_SIZE}  |  GradAcc: {GRAD_ACCUMULATION}")
    console.print(f"LR        : {LEARNING_RATE}  |  Warmup steps: {warmup_steps}\n")

    t0     = time.time()
    step   = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dl):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            images         = batch["image"]      # list numpy

            # Build pixel_values cho từng ảnh trong batch
            pixel_values = torch.cat(
                [model._build_pixel_values(img) for img in images], dim=0
            )

            # Forward — InternVL2 tính language modeling loss
            outputs = model.model(
                input_ids      = input_ids,
                attention_mask = attention_mask,
                pixel_values   = pixel_values,
                labels         = input_ids,   # causal LM: labels = input_ids
            )
            loss = outputs.loss / GRAD_ACCUMULATION
            loss.backward()

            epoch_loss += loss.item() * GRAD_ACCUMULATION

            if (batch_idx + 1) % GRAD_ACCUMULATION == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_trainable_params(), MAX_GRAD_NORM
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

        avg_loss = epoch_loss / max(len(dl), 1)
        console.print(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"loss={avg_loss:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

    console.print(f"\nTổng thời gian: {time.time() - t0:.1f}s")

    model.save_lora()
    console.print(f"[bold green]LoRA saved → {LORA_WEIGHTS_PATH}[/bold green]")

    if args.eval:
        _evaluate(model)


def _get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup + cosine decay."""
    import math
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(warmup_steps, 1)
        progress = float(current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def _evaluate(model: InternVL2Wrapper):
    """Đánh giá định tính: sinh caption và so sánh với ground truth."""
    console.print("\n[bold]Evaluation — Stage 3 (qualitative)[/bold]")

    ds = VLMDataset(augment=False)
    if not ds.samples:
        console.print("[red]Không có sample để evaluate.[/red]")
        return

    model.eval_mode()
    results = []

    for img_path, ann in ds.samples[:20]:   # Lấy tối đa 20 mẫu
        import numpy as np
        from PIL import Image
        image = np.array(Image.open(img_path).convert("RGB"))

        pred_caption = model.caption(image)
        gt_caption   = ann.get("caption", "")

        results.append({
            "image":      str(img_path),
            "gt":         gt_caption,
            "pred":       pred_caption,
            "defect_type": ann.get("defect_type"),
            "severity":    ann.get("severity"),
        })

        console.print(f"\n[cyan]{img_path.name}[/cyan]")
        console.print(f"GT  : {gt_caption}")
        console.print(f"Pred: {pred_caption}")

    out = Path(RESULTS_DIR) / "stage3_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    console.print(f"\nKết quả lưu → {out}")


def main():
    parser = argparse.ArgumentParser(description="Stage 3 — InternVL2 LoRA fine-tune")
    parser.add_argument("--category", default=CATEGORY)
    parser.add_argument("--eval",     action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
