# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 4-stage industrial fabric/garment defect detection pipeline. Images flow through anomaly detection → segmentation → VLM classification → rule-based decision, producing a Pass/Fail verdict with an optional HTML/JSON report.

## Environment

```bash
conda env create -f enviroment.yml
conda activate defect_detection
```

**Critical Windows constraints** (already applied — do not revert):
- `NUM_WORKERS = 0` in `configs/base_config.py` — multiprocessing DataLoader crashes on Windows CUDA
- `pin_memory = False` in all DataLoaders
- `sys.stdout.reconfigure(encoding='utf-8')` at top of any script that prints Vietnamese text

**Active model versions:**
- InternVL2-1B (not 8B) — fits RTX 3050 4GB with 4-bit quantization
- SAM2.1-Large — config key: `configs/sam2.1/sam2.1_hiera_l.yaml`
- DINOv2 ViT-L/14 — cached at `D:\MLCache\torch\hub\facebookresearch_dinov2_main`

## Common Commands

### Training (run sequentially — each stage depends on the previous)

```bash
python -m stage1_anomaly.train --category tsfabric_T1 --eval
python -m stage2_seg.train --category tsfabric_T1 --eval
python -m stage3_vlm.lora_train --category tsfabric_T1
```

### Inference

```bash
# Full pipeline (GPU machine)
python inference/run_pipeline.py --image <path> [--category tsfabric_T1] [--html]

# CPU-only machine (Ollama replaces Stage 3)
python inference/run_pipeline.py --image <path> --category tsfabric_T1 --use_ollama

# Stage 1 only (lightest, runs on any machine)
python inference/run_pipeline.py --image <path> --stage1_only
```

Exit code: `0` = Pass, `1` = Fail.

### Data Preparation

```bash
python scripts/prep_tsfabrics.py    # reads CSV with encoding="utf-8-sig" (BOM strip)
python scripts/split_tsfabrics.py
```

### Evaluation

```bash
python tools/evaluate.py
python tools/dashboard.py [--category <name>] [--watch]
```

## Architecture

The pipeline is orchestrated by `inference/run_pipeline.py`, which calls each stage in sequence and passes results forward.

### Stage 1 — Anomaly Detection (`stage1_anomaly/`)
- **Model:** DINOv2 ViT-L/14 frozen + PatchCore memory bank (FAISS IndexFlatL2, CPU-only via `faiss-cpu`)
- **Key classes:** `DINOv2Extractor` (hooks into transformer blocks), `MemoryBank` (coreset + k-NN), `PatchCore`
- **Coreset algorithm:** random pool (50K) → greedy coreset on pool — avoids O(N²) on full 495K patch set
- **Tuned config for tsfabric_T1:** `feature_layers=[12,18]`, `coreset_ratio=0.02`, `max_samples=10000`, `k_nearest=9`, `percentile=99.0`
- **Checkpoint:** `outputs/checkpoints/stage1_{category}_bank.npy` + `_meta.json`

### Stage 2 — Defect Segmentation (`stage2_seg/`)
- **Model:** SAM2.1-Large with LoRA (rank 16, alpha 32) on `q_proj, v_proj, k_proj, out_proj`
- **Training fix:** `SAM2Wrapper.encode_for_training()` bypasses `set_image()` (`@no_grad`) — calls `model.forward_image()` directly so LoRA gradients flow through the image encoder
- **Prompting:** Top-8 anomaly points + bounding box from Stage 1 score map (threshold 0.4)
- **Config:** `TRAIN_BATCH_SIZE=1`, `NUM_EPOCHS=10`, `SAM2_CONFIG="configs/sam2.1/sam2.1_hiera_l.yaml"`

### Stage 3 — VLM Classification (`stage3_vlm/`)
- **Model:** InternVL2-1B with LoRA only on `language_model` (not full model — wrapping full InternVLChatModel with PEFT causes `inputs_embeds` TypeError)
- **Quantization:** 4-bit (BitsAndBytesConfig, `bnb_4bit_compute_dtype=torch.float16`) during training; `force_fp16=True` during inference to avoid dtype mismatch
- **Cached model fix:** `modeling_internvl_chat.py` line ~334 must cast `vit_embeds` dtype: `.to(dtype=input_embeds.dtype, device=input_embeds.device)`
- **Alternate inference:** `stage3_vlm/inference_ollama.py` — uses Ollama HTTP API (`moondream` or `llava:7b-q4_0`), same interface as `VLMInference`, for CPU-only machines

### Stage 4 — Decision Engine (`stage4_decision/`)
- **ALWAYS_FAIL types:** `hole`, `tear`, `contamination`
- **MAJOR_FAIL types:** `stain`, `weave`, `discoloration`, `yarn`
- Reports via `stage4_decision/report.py` (JSON, HTML, console)

### Datasets
- **TSfabrics** (primary): 93K industrial fabric video frames, categories T1/T2/T3
- Layout after `split_tsfabrics.py`: `data/pass/train|test|eval/<category>/` and `data/fail/train|test/<category>/` with masks at `data/fail/train/masks/<category>/` and `data/fail/masks/<category>/`

## Configuration

`configs/base_config.py` — device, paths, `NUM_WORKERS=0`  
`configs/stage{1..4}_config.py` — per-stage hyperparameters and model paths  
`configs/stage1_config.py` — `category_tuning` dict with per-fabric `feature_layers`, `k_nearest`, `threshold_percentile`

**Checkpoint files needed for inference:**
```
outputs/checkpoints/
  stage1_{category}_bank.npy      # memory bank vectors
  stage1_{category}_meta.json     # threshold + embed_dim
  stage2_{category}_lora.pt       # SAM2 LoRA weights
  stage3_{category}_lora/         # InternVL2 adapter (safetensors + tokenizer)
weights/
  internvl2/                      # InternVL2-1B base model
  sam2/sam2_hiera_large.pt        # SAM2.1 checkpoint
```
