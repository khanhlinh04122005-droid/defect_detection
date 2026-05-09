# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 4-stage industrial fabric/garment defect detection pipeline. Images flow through anomaly detection → segmentation → VLM classification → rule-based decision, producing a Pass/Fail verdict with an optional HTML/JSON report.

## Environment Setup

```bash
conda env create -f enviroment.yml
```

Weights, data, and outputs are excluded from git (see `.gitignore`). Use `python scripts/dl_weights.py` to download pre-trained weights.

## Common Commands

### Run Inference

```bash
# Single image
python inference/run_pipeline.py --image <path> [--category tsfabric_T1] [--device cuda] [--output <dir>] [--html]

# Batch
python inference/batch_runner.py --input_dir <dir> --output <dir>

# Quickstart demo
python scripts/quickstart.py
```

Exit code: `0` = Pass, `1` = Fail.

### Data Preparation

```bash
python scripts/prep_tsfabrics.py    # Prepare TSfabrics dataset
python scripts/split_tsfabrics.py   # Train/test split
python scripts/prep_mvtec.py        # Prepare MVTec-AD dataset
```

### Training

```bash
python stage1_anomaly/train.py      # PatchCore memory bank (no gradient training)
python stage2_seg/train.py          # SAM2 LoRA fine-tune (needs ≥20 samples)
python stage3_vlm/lora_train.py     # InternVL2 LoRA fine-tune (needs ≥50 samples)
```

### Evaluation & Tooling

```bash
python tools/evaluate.py            # Compute metrics
python tools/dashboard.py [--category <name>] [--watch]
python tools/annotate.py
```

## Architecture

The pipeline is orchestrated by `inference/run_pipeline.py`, which calls each stage in sequence and passes results forward.

### Stage 1 — Anomaly Detection (`stage1_anomaly/`)
- **Model:** DINOv2 ViT-L/14 as frozen feature extractor + PatchCore memory bank
- **Key classes:** `DINOv2Extractor`, `MemoryBank` (coreset sampling + k-NN search), `PatchCore`
- **Output:** Pixel-level anomaly score map + binary flag
- **Config:** `configs/stage1_config.py` — per-category thresholds (98.5–99.5 percentile), k-NN=9–11, feature layers `[9, 12, 15, 18]`

### Stage 2 — Defect Segmentation (`stage2_seg/`)
- **Model:** SAM2-Large with optional LoRA (rank 16, alpha 32)
- **Prompting:** Top-8 anomaly points + bounding box from Stage 1 map (threshold 0.4)
- **Output:** Binary defect mask + confidence
- **Config:** `configs/stage2_config.py`

### Stage 3 — VLM Classification (`stage3_vlm/`)
- **Model:** InternVL2-8B (HuggingFace) with optional LoRA on attention + FFN layers
- **Output:** Defect type, severity level, textual description
- **Config:** `configs/stage3_config.py` — temperature 0.2, top-p 0.9, batch=2 with grad_accumulation=8

### Stage 4 — Decision Engine (`stage4_decision/`)
- **Method:** Rule-based logic in `stage4_decision/rules.py`
- **ALWAYS_FAIL types:** missing, crack, contamination
- **MAJOR_FAIL types:** scratch, dent, discoloration, burr
- **Output:** Final Pass/Fail verdict with severity ranking; reports via `stage4_decision/report.py` (JSON, HTML, console)

### Data Pipeline (`data_pipeline/`)
- `augmentation.py` — standard augmentations
- `cutpaste.py` — CutPaste synthetic defect generation
- `synthetic_gen.py` — broader synthetic defect generation

### Datasets
- **TSfabrics** (primary): 93K industrial fabric video frames, categories T1/T2/T3
- **MVTec-AD** (legacy support): standard industrial surface defect benchmark
- Per-category tuning parameters live in `configs/stage1_config.py` and `configs/base_config.py`

## Configuration

All stage configs inherit from `configs/base_config.py`:
- Device: auto CUDA/CPU
- Standard image size: 224×224, ImageNet normalization
- Directory layout: `data/`, `weights/`, `outputs/`, `checkpoints/`

Per-stage configs (`configs/stage{1..4}_config.py`) hold model paths, hyperparameters, and per-category overrides.
