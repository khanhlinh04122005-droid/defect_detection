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
    "You are a fabric quality inspector. "
    "Look at the image and describe the defect briefly: type, location, and severity. "
    "Answer in 1-2 sentences only."
)

# VQA questions in English — InternVL2-1B follows English instructions more reliably
VQA_QUESTIONS = {
    "has_defect"   : "Does this fabric have a visible defect? Answer with exactly one word: yes or no.",

    "defect_type"  : (
        "What type of fabric defect is shown? "
        "Answer with exactly one word from this list: "
        "hole / tear / stain / yarn / weave / pilling / discoloration / contamination / other."
    ),

    "severity"     : "How severe is the defect? Answer with exactly one word: minor / major / critical.",

    "location"     : "Where is the defect located? Answer briefly, e.g.: top-left / center / right edge.",

    "size_estimate": "What percentage of the fabric area is affected by the defect? Answer with a number only, e.g.: 5%.",

    "pass_fail"    : "Does this fabric pass quality control? Answer with exactly one word: Pass or Fail.",
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