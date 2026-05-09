import torch
import os

# =====================================================================
# FABRIC & GARMENT DEFECT DETECTION
# Hệ thống phát hiện lỗi vải và sản phẩm may mặc
# =====================================================================

# ── Nguồn dữ liệu ──────────────────────────────────────────────────
# Mỗi category ánh xạ tới nguồn dataset tương ứng
# Cho phép dùng song song MVTec + TSfabrics mà không cần trộn chung

# ── Chỉ dùng TSfabrics ─────────────────────────────────────────────
# MVTec đã được loại bỏ khỏi workflow chính.
# Nếu muốn bật lại MVTec, thêm lại các entry vào DATASET_SOURCES.
DATASET_SOURCES = {
    # ── TSfabrics dataset (93K ảnh vải công nghiệp thực tế) ────────

    # ── TSfabrics (video-frame, vải công nghiệp thực tế) ─────────
    "tsfabric_T1": "tsfabrics",   # Loại vải T1 (dệt thoi)
    "tsfabric_T2": "tsfabrics",   # Loại vải T2 (dệt thoi khác)
    "tsfabric_T3": "tsfabrics",   # Loại vải T3 (dệt thoi khác)

    # ── Vải tự thu thập (custom) ──────────────────────────────────
    "cotton"    : "custom",
    "denim"     : "custom",
    "silk"      : "custom",
    "synthetic" : "custom",
    "wool"      : "custom",
    "knit"      : "custom",
}

# Tất cả các categories được hỗ trợ
FABRIC_CATEGORIES = list(DATASET_SOURCES.keys())

# Category đang chạy (đổi tại đây hoặc dùng --category khi chạy CLI)
CATEGORY = "tsfabric_T1"   # Default: dùng TSfabrics loại vải T1

# ── Đường dẫn dataset ───────────────────────────────────────────────
TSFABRICS_DIR  = "data/TSfabrics"       # TSfabrics root (dataset chính)
MVTEC_DIR      = "data/mvtec"           # MVTec-AD (không dùng, giữ để tham khảo)

DATA_DIR    = "data"
PASS_DIR    = "data/pass"
FAIL_DIR    = "data/fail"
VLM_ANN_DIR = "data/vlm_ann"

FAIL_IMAGES_DIR = "data/fail/images"
FAIL_MASKS_DIR  = "data/fail/masks"
ANNOTATED_DIR   = "data/annotated"
PASS_TRAIN_DIR  = "data/pass/train"
PASS_TEST_DIR   = "data/pass/test"

# ── Đường dẫn weights và outputs ────────────────────────────────────
WEIGHTS_DIR    = "weights"
OUTPUT_DIR     = "outputs"
RESULTS_DIR    = "outputs/results"
EVAL_DIR       = "outputs/eval"
CHECKPOINT_DIR = "outputs/checkpoints"
# Mỗi category có checkpoint riêng → train song song không bị ghi đè
# Ví dụ: outputs/checkpoints/stage1_carpet/  và  outputs/checkpoints/stage1_tsfabric_T1/

# ── Hardware & Seed ─────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0   # Windows: luôn 0 để tránh crash DataLoader multiprocessing
PIN_MEMORY  = True
SEED        = 42
IMAGE_SIZE  = (224, 224)
IMAGE_MEAN  = (0.485, 0.456, 0.406)
IMAGE_STD   = (0.229, 0.224, 0.225)

LOG_LEVEL = "INFO"
LOG_DIR   = "outputs/logs"

# ── Helper ──────────────────────────────────────────────────────────
def get_source(category: str) -> str:
    """Trả về tên nguồn dataset cho một category."""
    return DATASET_SOURCES.get(category, "custom")

def make_dirs():
    dirs = [
        DATA_DIR, PASS_DIR, FAIL_DIR, VLM_ANN_DIR,
        WEIGHTS_DIR, RESULTS_DIR, EVAL_DIR,
        CHECKPOINT_DIR, LOG_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    make_dirs()
    print("Đã tạo toàn bộ thư mục dự án.")
    print(f"Device  : {DEVICE}")
    print(f"Category: {CATEGORY} (source: {get_source(CATEGORY)})")
    print(f"Supported categories ({len(FABRIC_CATEGORIES)}): {FABRIC_CATEGORIES}")