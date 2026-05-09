# AGENTS.md

This file provides guidance to AI coding agents working in this repository.

## Project Overview

A 4-stage industrial fabric defect detection pipeline:
1. **Stage 1** — PatchCore anomaly detection (DINOv2 + FAISS)
2. **Stage 2** — SAM2 defect segmentation (LoRA fine-tuned)
3. **Stage 3** — InternVL2-1B VLM classification (LoRA fine-tuned)
4. **Stage 4** — Rule-based decision engine → Pass/Fail verdict

## Environment

```bash
conda activate defect_detection
set PYTHONPATH=D:\Projects\defect_detection   # Windows
export PYTHONPATH=/content/defect_detection   # Linux/Colab
```

## Run Inference

```bash
python inference/run_pipeline.py --image <path> --category tsfabric_T1 [--html]
python inference/run_pipeline.py --image <path> --stage1_only
python app.py [--share] [--ollama] [--category tsfabric_T1]
```

## Training

```bash
python -m stage1_anomaly.train --category tsfabric_T1 --eval
python -m stage2_seg.train --category tsfabric_T1 --eval
python -m stage3_vlm.lora_train --category tsfabric_T1
python scripts/tune_threshold.py --category tsfabric_T1 [--apply]
```

## Architecture

### Pipeline flow
`run_pipeline.py` → `PipelineEngine` (`stage4_decision/engine.py`) → lazy-loads each stage → `decide()` (`stage4_decision/rules.py`)

### Stage 1 (`stage1_anomaly/`)
- `patchcore.py`: main class, `predict()` returns `{image_score, threshold, prediction, score_map}`
- `memory_bank.py`: FAISS index + greedy coreset (`GREEDY_LIMIT=50000`, dot-product distance)
- `extractor.py`: DINOv2 hook-based feature extraction

### Stage 2 (`stage2_seg/`)
- `sam2_wrapper.py`: `predict_mask()` for inference, `encode_for_training()` bypasses `@no_grad` for LoRA training
- `lora_adapter.py`: inject/save/load LoRA into SAM2 Hiera (`qkv`, `proj` modules)
- `train.py`: bfloat16 autocast + per-image backward (batch_size=1)

### Stage 3 (`stage3_vlm/`)
- `model.py`: `InternVL2Wrapper` — LoRA on `language_model` only (not full model)
- `inference.py`: `VLMInference` — GPU path (InternVL2)
- `inference_ollama.py`: `VLMInferenceOllama` — CPU path (Ollama HTTP API)

### Stage 4 (`stage4_decision/`)
- `engine.py`: `PipelineEngine` — lazy load, unloads SAM2 before InternVL2 to avoid OOM
- `rules.py`: weighted voting (Stage1=0.6, Stage2=0.2, Stage3=0.2)
- `report.py`: JSON/HTML/console output

### UI (`app.py`)
- Gradio `gr.Blocks` with image upload + pipeline result + chat
- Chat uses pipeline result context + image for follow-up Q&A
- `type="messages"` format for `gr.Chatbot`

## Critical Constraints

- `NUM_WORKERS=0` — Windows CUDA DataLoader
- SAM2 LoRA targets: `["qkv", "proj"]` (NOT `q_proj/v_proj`)
- `device_map={"": 0}` with bitsandbytes, `None` + `.to(device)` without
- `modeling_internvl_chat.py` line ~334: `.to(dtype=input_embeds.dtype, device=input_embeds.device)`
- Stage 1 voting weight 0.6 (most reliable when Stage 3 LoRA not fully trained)

## Colab

- Notebook: `scripts/colab_run.ipynb`
- Pack data: `python scripts/pack_for_colab.py` → upload `code.zip` + `data_tsfabric_T1.zip` to Drive
- Pin: `transformers==4.44.0`, `torchao>=0.16.0`
- Auto-patch cell fixes all Colab compatibility issues each session

## Checkpoint Files

```
outputs/checkpoints/
  stage1_{category}_bank.npy + _meta.json
  stage2_{category}_lora.pt          # optional
  stage3_{category}_lora/            # adapter_model.safetensors + tokenizer
weights/
  sam2/sam2_hiera_large.pt
```
