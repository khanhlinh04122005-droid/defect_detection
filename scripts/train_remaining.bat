@echo off
echo ===== Train Stage 1 + 2 =====
cd /d D:\Projects\defect_detection
call conda activate defect_detection

echo [1/3] Download SAM2 weights...
python scripts/dl_weights.py --sam2

echo [2/3] Train Stage 1 - PatchCore...
python -m stage1_anomaly.train --category tsfabric_T1 --eval

echo [3/3] Train Stage 2 - SAM2 LoRA...
python -m stage2_seg.train --category tsfabric_T1

echo ===== DONE =====
pause
