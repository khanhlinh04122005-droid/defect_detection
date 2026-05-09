from configs.base_config import WEIGHTS_DIR, CHECKPOINT_DIR, CATEGORY

VLM_MODEL_NAME   = "OpenGVLab/InternVL2-1B"   # 1B: ~2GB VRAM, phù hợp RTX 3050 4GB
VLM_WEIGHTS      = f"{WEIGHTS_DIR}/internvl2"
LOAD_IN_8BIT     = False
LOAD_IN_4BIT     = True        # 4-bit quantization cho 4GB VRAM

LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
    "gate_proj", "up_proj", "down_proj",        # FFN
]


TRAIN_BATCH_SIZE    = 1        # RTX 3050 4GB: batch=1
GRAD_ACCUMULATION   = 16       # Effective batch = 1 × 16 = 16
LEARNING_RATE       = 2e-5
NUM_EPOCHS          = 10
WARMUP_RATIO        = 0.1
WEIGHT_DECAY        = 0.01
MAX_GRAD_NORM       = 1.0

# Số ảnh tối thiểu để fine-tune VLM
MIN_CAPTIONED_SAMPLES = 50     # Dưới 50: dùng zero-shot với prompt template


# Mỗi ảnh trong vlm_ann/ cần 1 file JSON cùng tên, ví dụ:
# data/vlm_ann/img_001.jpg → data/vlm_ann/img_001.json
# Format JSON:
# {
#   "caption": "Vết xước ngang 3mm ở góc trên bên phải, mức độ nhẹ",
#   "defect_type": "scratch",
#   "severity": "minor",
#   "location": "top-right",
#   "pass_fail": "fail"
# }
ANNOTATION_FORMAT   = "json"
CAPTION_MAX_LENGTH  = 128      # Độ dài caption tối đa (token)
MAX_NEW_TOKENS      = 256      # Số token tối đa model sinh ra
TEMPERATURE         = 0.2      # Thấp = ổn định, cao = sáng tạo hơn
TOP_P               = 0.9
DO_SAMPLE           = False    # False = greedy (ổn định hơn cho production)

# System prompt chuyên ngành kiểm tra chất lượng vải / may mặc
SYSTEM_PROMPT_CAPTION = (
    "Bạn là chuyên gia kiểm tra chất lượng vải trong ngành may mặc công nghiệp. "
    "Nhị n vào ảnh, hãy mô tả ngắn gọn loại lỗi trên vải, vị trí xuất hiện và mức độ ảnh hưởng "
    "đến chất lượng sản phẩm. "
    "Trả lời bằng tiếng Việt, tối đa 2 câu, sử dụng thuật ngữ may mặc chuẩn."
)

# VQA dành riêng cho vải / may mặc
VQA_QUESTIONS = {
    "has_defect"   : "Vải trong ảnh có lỗi không? Chỉ trả lời đúng 1 từ: Có hoặc Không.",

    "defect_type"  : (
        "Loại lỗi vải trong ảnh là gì? "
        "Chỉ trả lời đúng 1 từ trong danh sách sau, không giải thích: "
        "hole / tear / stain / yarn / weave / pilling / discoloration / contamination / other."
    ),

    "severity"     : (
        "Mức độ lỗi vải là gì? "
        "Chỉ trả lời đúng 1 từ: minor / major / critical."
    ),

    "location"     : (
        "Lỗi nằm ở đâu trên tấm vải? "
        "Chỉ trả lời đúng 1 cụm từ ngắn, ví dụ: góc trên trái / giữa / biên phải."
    ),

    "size_estimate": (
        "Vùng lỗi chiếm bao nhiêu % diện tích vải? "
        "Chỉ trả lời số phần trăm, ví dụ: 5%."
    ),

    "pass_fail"    : (
        "Vải này có đạt tiêu chuẩn xuất xưởng không? "
        "Chỉ trả lời đúng 1 từ: Pass hoặc Fail."
    ),
}

LORA_WEIGHTS_PATH   = f"{CHECKPOINT_DIR}/stage3_{CATEGORY}_lora"
FULL_MODEL_PATH     = f"{CHECKPOINT_DIR}/stage3_{CATEGORY}_full"

if __name__ == "__main__":
    print("=== Stage 3 Config (InternVL2-8B) ===")
    print(f"Model        : {VLM_MODEL_NAME}")
    print(f"LoRA rank    : {LORA_RANK}")
    print(f"Batch size   : {TRAIN_BATCH_SIZE} (eff. {TRAIN_BATCH_SIZE * GRAD_ACCUMULATION})")
    print(f"Min samples  : {MIN_CAPTIONED_SAMPLES} ảnh có caption")
    print(f"VQA questions: {list(VQA_QUESTIONS.keys())}")
    print(f"LoRA path    : {LORA_WEIGHTS_PATH}")