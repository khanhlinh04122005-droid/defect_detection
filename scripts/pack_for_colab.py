"""
scripts/pack_for_colab.py

Đóng gói code + data để upload lên Google Drive cho Colab.
Chạy trên máy local:
    python scripts/pack_for_colab.py

Tạo ra 2 file zip tại thư mục gốc dự án:
    code.zip             (~2 MB)  — source code, không gồm data/weights/outputs
    data_tsfabric_T1.zip (~312 MB) — data train cần thiết
"""

import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent.parent

# ── 1. Code zip ───────────────────────────────────────────────────────────────
CODE_ZIP = ROOT / "code.zip"
CODE_EXCLUDES = {
    "data", "weights", "outputs", "__pycache__",
    ".git", ".venv", "node_modules", "*.pyc",
}

def should_exclude(path: Path) -> bool:
    for part in path.parts:
        if part in CODE_EXCLUDES or part.endswith(".pyc"):
            return True
    return False

print("Đang tạo code.zip...")
with zipfile.ZipFile(CODE_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    count = 0
    for f in ROOT.rglob("*"):
        if f.is_file() and not should_exclude(f.relative_to(ROOT)):
            arcname = "defect_detection/" + str(f.relative_to(ROOT))
            zf.write(f, arcname)
            count += 1

size_mb = CODE_ZIP.stat().st_size / 1e6
print(f"✓ code.zip — {count} files, {size_mb:.1f} MB")

# ── 2. Data zip ───────────────────────────────────────────────────────────────
DATA_ZIP = ROOT / "data_tsfabric_T1.zip"
CATEGORY = "tsfabric_T1"

data_dirs = [
    (ROOT / "data" / "pass" / "train" / CATEGORY, f"defect_detection/data/pass/train/{CATEGORY}"),
    (ROOT / "data" / "fail" / "train" / CATEGORY, f"defect_detection/data/fail/train/{CATEGORY}"),
    (ROOT / "data" / "fail" / "masks" / CATEGORY, f"defect_detection/data/fail/masks/{CATEGORY}"),
    (ROOT / "data" / "vlm_ann",                   "defect_detection/data/vlm_ann"),
]

print("\nĐang tạo data_tsfabric_T1.zip...")
with zipfile.ZipFile(DATA_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    total = 0
    for src_dir, arc_prefix in data_dirs:
        if not src_dir.exists():
            print(f"  ⚠ Không tìm thấy: {src_dir}")
            continue
        files = list(src_dir.rglob("*.*"))
        for f in files:
            arcname = arc_prefix + "/" + str(f.relative_to(src_dir))
            zf.write(f, arcname)
        total += len(files)
        print(f"  + {arc_prefix.split('/')[-1]}: {len(files)} files")

size_mb = DATA_ZIP.stat().st_size / 1e6
print(f"✓ data_tsfabric_T1.zip — {total} files, {size_mb:.1f} MB")

# ── Hướng dẫn ─────────────────────────────────────────────────────────────────
print(f"""
══════════════════════════════════════════════
Bước tiếp theo:
1. Upload 2 file lên Google Drive:
   - {CODE_ZIP.name}  →  Drive/defect_detection/
   - {DATA_ZIP.name}  →  Drive/defect_detection/

2. Mở Google Colab, upload notebook:
   scripts/colab_run.ipynb

3. Chạy từng cell theo thứ tự.
   Cell 9 sẽ tạo link Gradio public để demo.
══════════════════════════════════════════════
""")
