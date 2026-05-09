#!/bin/bash
# Chạy toàn bộ pipeline stage3: prep → split → train
set -e
cd "$(dirname "$0")/.."

CONDA_RUN="conda run -n defect_detection"

# Bước 1: Kiểm tra data TSfabrics
if [ ! -d "data/TSfabrics/IMAGE" ]; then
  echo "ERROR: data/TSfabrics/IMAGE chưa tồn tại. Hãy giải nén TSfabrics.zip trước."
  exit 2
fi

# Bước 2: Prep nếu chưa có fail images
if [ ! -d "data/fail/images" ] || [ -z "$(ls -A data/fail/images 2>/dev/null)" ]; then
  echo ">>> Chạy prep_tsfabrics.py..."
  $CONDA_RUN python scripts/prep_tsfabrics.py
fi

# Bước 3: Split nếu chưa có vlm_ann
ANN_COUNT=$(find data/vlm_ann -name "*.json" 2>/dev/null | wc -l)
if [ "$ANN_COUNT" -lt 50 ]; then
  echo ">>> Chạy split_tsfabrics.py (hiện có $ANN_COUNT annotations)..."
  $CONDA_RUN python scripts/split_tsfabrics.py
fi

# Bước 4: Train
echo ">>> Chạy stage3 training..."
$CONDA_RUN python -m stage3_vlm.lora_train --category tsfabric_T1

echo "DONE"
