from configs.base_config import WEIGHTS_DIR, CHECKPOINT_DIR, CATEGORY

# SAM2 backbone
SAM2_MODEL_SIZE  = "large"     # "tiny" | "small" | "base_plus" | "large"
                               # large = tốt nhất, vẫn chạy được trên 3090/4090
SAM2_WEIGHTS     = f"{WEIGHTS_DIR}/sam2/sam2_hiera_large.pt"
SAM2_CONFIG      = "sam2_hiera_l.yaml"   # Config đi kèm SAM2

# LoRA adapter
LORA_RANK        = 16          # Rank của LoRA (8–32 là hợp lý)
LORA_ALPHA       = 32          # Scale factor = alpha / rank
LORA_DROPOUT     = 0.1
# Module nào trong SAM2 được inject LoRA
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "out_proj"]


# Stage 1 trả về anomaly score map → Stage 2 dùng làm prompt cho SAM2
# Vải có texture phân tán hơn điện tử → cần TOPK cao hơn và threshold thấp hơn
PROMPT_TYPE      = "both"      # "point" | "box" | "both" — "both" tốt hơn cho vải
TOPK_POINTS      = 8           # Lấy 8 điểm (vải có lỗi trải rộng hơn kim loại)
POINT_LABEL      = 1           # 1 = foreground (vùng lỗi), 0 = background

# Ngưỡng anomaly score để lọc điểm prompt
# Giảm xuống 0.4 vì texture vải có biến thiên tự nhiên cao hơn bề mặt cứng
PROMPT_THRESHOLD = 0.40        # Chỉ lấy điểm có score > 40%

# Fine-tune LoRA
TRAIN_BATCH_SIZE = 4           # Nhỏ vì SAM2-large tốn VRAM
LEARNING_RATE    = 1e-4
NUM_EPOCHS       = 30
WARMUP_EPOCHS    = 3
WEIGHT_DECAY     = 0.01

# Số ảnh fail có mask tối thiểu để bắt đầu fine-tune
MIN_LABELED_SAMPLES = 20       # Dưới 20 ảnh: dùng SAM2 zero-shot, chưa fine-tune

# Loss
DICE_LOSS_WEIGHT = 0.5
BCE_LOSS_WEIGHT  = 0.5

# Augmentation cho fine-tune SAM2
AUG_FLIP         = True
AUG_ROTATION     = 15          # Độ xoay tối đa (±15°)
AUG_SCALE        = (0.8, 1.2)  # Scale range

# Đường dẫn
LORA_WEIGHTS_PATH   = f"{CHECKPOINT_DIR}/stage2_{CATEGORY}_lora.pt"
FULL_MODEL_PATH     = f"{CHECKPOINT_DIR}/stage2_{CATEGORY}_full.pt"

# Evaluation targets
TARGET_MIOU      = 0.80        # mIoU mục tiêu cho segmentation mask
TARGET_DICE      = 0.85

if __name__ == "__main__":
    print("=== Stage 2 Config (SAM2 + LoRA) ===")
    print(f"SAM2 model   : {SAM2_MODEL_SIZE}")
    print(f"LoRA rank    : {LORA_RANK}")
    print(f"Prompt type  : {PROMPT_TYPE} (top-{TOPK_POINTS} points)")
    print(f"Min samples  : {MIN_LABELED_SAMPLES} ảnh có mask")
    print(f"LoRA weights : {LORA_WEIGHTS_PATH}")