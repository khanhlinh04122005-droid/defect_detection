"""
scripts/prep_tsfabrics.py

Chuyển đổi dataset TSfabrics sang cấu trúc chuẩn của pipeline.

Cấu trúc TSfabrics:
  data/TSfabrics/
    IMAGE/<sequence>/000001.jpeg  ...
    CSV/<sequence>.csv            → FileName, Label (1=pass, 2/4/5=fail)
    ANNOTATION/<sequence>/defect/ → Polygon annotations

Cấu trúc output (chuẩn pipeline):
  data/pass/train/tsfabric_T<x>/   → Ảnh label=1 (train PatchCore)
  data/pass/test/tsfabric_T<x>/    → Ảnh label=1 (tính threshold)
  data/fail/images/tsfabric_T<x>/  → Ảnh label=2/4/5
  data/fail/masks/tsfabric_T<x>/   → Mask PNG (nếu có annotation)

Label mapping:
  1 → pass (good)
  2 → fail (minor defect)
  4 → fail (weave/thread defect)
  5 → fail (severe defect)

Cách chạy:
  python scripts/prep_tsfabrics.py
  python scripts/prep_tsfabrics.py --fabric_type T1  # Chỉ xử lý T1
  python scripts/prep_tsfabrics.py --train_ratio 0.8 --val_ratio 0.1
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

from PIL import Image

# ── Label mapping ────────────────────────────────────────────────────
PASS_LABELS = {1}        # Ảnh vải bình thường
FAIL_LABELS = {2, 4, 5}  # Ảnh vải có lỗi

DEFECT_TYPE_MAP = {
    2: "thread",     # Lỗi sợi / đường chỉ nhẹ
    4: "weave",      # Lỗi dệt (phổ biến nhất)
    5: "hole",       # Lỗi thủng / nghiêm trọng
}

SEVERITY_MAP = {
    2: "minor",
    4: "major",
    5: "critical",
}


def get_fabric_type(seq_name: str) -> str:
    """Lấy loại vải từ tên sequence: T1_S148_I108_1 → T1"""
    return seq_name.split("_")[0]  # T1, T2, hoặc T3


def prep_sequence(
    seq_name: str,
    tsfabrics_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    sample_stride: int = 5,
) -> dict:
    """
    sample_stride: Chỉ lấy 1 frame trong mỗi N frame liên tiếp.
    Ví dụ stride=5 → lấy frame 1, 6, 11, 16, ...
    Giúp loại bỏ frame trùng lặp trong video-frame sequences.
    Mặc định stride=5 → từ 75K ảnh pass còn lại ~15K ảnh đủ đa dạng.
    """
    image_dir = tsfabrics_dir / "IMAGE" / seq_name
    csv_file  = tsfabrics_dir / "CSV" / f"{seq_name}.csv"
    ann_dir   = tsfabrics_dir / "ANNOTATION" / seq_name / "defect"

    if not image_dir.exists() or not csv_file.exists():
        print(f"  [SKIP] {seq_name}: Không tìm thấy ảnh hoặc CSV")
        return {}

    fabric_type = get_fabric_type(seq_name)         # T1 / T2 / T3
    category    = f"tsfabric_{fabric_type}"         # tsfabric_T1

    # Đường dẫn output
    pass_train = output_dir / "pass" / "train" / category
    pass_val   = output_dir / "pass" / "test"  / category
    fail_img   = output_dir / "fail" / "images" / category
    fail_mask  = output_dir / "fail" / "masks"  / category
    vlm_ann    = output_dir / "vlm_ann"

    for d in [pass_train, pass_val, fail_img, fail_mask, vlm_ann]:
        d.mkdir(parents=True, exist_ok=True)

    # Đọc CSV labels
    rows = []
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["FileName"], int(row["Label"])))

    # Áp dụng stride sampling lên pass frames để tránh trùng lặp
    pass_frames  = [(fn, lb) for i, (fn, lb) in enumerate(rows)
                    if lb in PASS_LABELS and i % sample_stride == 0]
    fail_frames  = [(fn, lb) for fn, lb in rows if lb in FAIL_LABELS]
    # Ảnh fail KHÔNG sample (giữ tất cả vì tỷ lệ fail đã thấp)

    # Chia pass thành train / val
    n_train = int(len(pass_frames) * train_ratio)
    n_val   = int(len(pass_frames) * val_ratio)
    train_frames = pass_frames[:n_train]
    val_frames   = pass_frames[n_train: n_train + n_val]

    stats = {"pass_train": 0, "pass_val": 0, "fail": 0, "ann_json": 0}

    # Copy pass/train
    for fn, _ in train_frames:
        # CSV dùng .png nhưng IMAGE folder có .jpeg
        src = image_dir / fn.replace(".png", ".jpeg")
        if not src.exists():
            src = image_dir / fn
        if src.exists():
            shutil.copy2(src, pass_train / src.name)
            stats["pass_train"] += 1

    # Copy pass/val (threshold)
    for fn, _ in val_frames:
        src = image_dir / fn.replace(".png", ".jpeg")
        if not src.exists():
            src = image_dir / fn
        if src.exists():
            shutil.copy2(src, pass_val / src.name)
            stats["pass_val"] += 1

    # Copy fail + tạo VLM annotation JSON
    for fn, label in fail_frames:
        src = image_dir / fn.replace(".png", ".jpeg")
        if not src.exists():
            src = image_dir / fn
        if not src.exists():
            continue

        dst_img = fail_img / src.name
        shutil.copy2(src, dst_img)
        stats["fail"] += 1

        # Tạo mask trắng đen từ annotation nếu có
        stem = Path(fn).stem
        mask_created = False
        if ann_dir.exists():
            ann_file = ann_dir / f"{stem}.png"
            if ann_file.exists():
                shutil.copy2(ann_file, fail_mask / f"{stem}.png")
                mask_created = True

        # Tạo VLM annotation JSON
        ann_json = {
            "image":       str(dst_img),
            "mask":        str(fail_mask / f"{stem}.png") if mask_created else None,
            "caption":     (
                f"Phát hiện lỗi vải công nghiệp loại {DEFECT_TYPE_MAP.get(label, 'other')} "
                f"trên cuộn vải {fabric_type}. "
                f"Mức độ: {SEVERITY_MAP.get(label, 'major')}."
            ),
            "defect_type": DEFECT_TYPE_MAP.get(label, "other"),
            "severity":    SEVERITY_MAP.get(label, "major"),
            "source":      "tsfabrics",
            "sequence":    seq_name,
            "label_id":    label,
        }
        json_path = vlm_ann / f"{seq_name}_{stem}.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(ann_json, jf, ensure_ascii=False, indent=2)
        stats["ann_json"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị TSfabrics dataset cho pipeline phát hiện lỗi vải"
    )
    parser.add_argument(
        "--tsfabrics_dir", default="data/TSfabrics",
        help="Thư mục gốc TSfabrics (mặc định: data/TSfabrics)"
    )
    parser.add_argument(
        "--output_dir", default="data",
        help="Thư mục output chuẩn pipeline (mặc định: data/)"
    )
    parser.add_argument(
        "--fabric_type", default=None,
        help="Chỉ xử lý loại vải cụ thể: T1, T2, T3 (mặc định: tất cả)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="Tỷ lệ ảnh pass dùng để train (mặc định: 0.8)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1,
        help="Tỷ lệ ảnh pass dùng để tính threshold (mặc định: 0.1)"
    )
    parser.add_argument(
        "--sample_stride", type=int, default=5,
        help="Lấy 1 frame mỗi N frame liên tiếp cho ảnh pass (mặc định: 5)"
    )
    args = parser.parse_args()

    tsfabrics_dir = Path(args.tsfabrics_dir)
    output_dir    = Path(args.output_dir)
    image_root    = tsfabrics_dir / "IMAGE"

    if not image_root.exists():
        print(f"[ERROR] Không tìm thấy: {image_root}")
        return

    sequences = sorted(d.name for d in image_root.iterdir() if d.is_dir())

    # Lọc theo fabric_type nếu có
    if args.fabric_type:
        prefix = args.fabric_type.upper()
        sequences = [s for s in sequences if s.startswith(prefix)]

    print(f"\n{'='*55}")
    print(f"  TSfabrics → Pipeline Data Preparation")
    print(f"{'='*55}")
    print(f"  Source   : {tsfabrics_dir}")
    print(f"  Output   : {output_dir}")
    print(f"  Sequences: {len(sequences)}")
    print(f"{'='*55}\n")

    total = {"pass_train": 0, "pass_val": 0, "fail": 0, "ann_json": 0}

    for seq in sequences:
        fabric_type = get_fabric_type(seq)
        category    = f"tsfabric_{fabric_type}"
        print(f"[{seq}]  →  category: {category}")

        stats = prep_sequence(
            seq_name      = seq,
            tsfabrics_dir = tsfabrics_dir,
            output_dir    = output_dir,
            train_ratio   = args.train_ratio,
            val_ratio     = args.val_ratio,
            sample_stride = args.sample_stride,
        )

        if stats:
            print(f"  pass/train: {stats['pass_train']}  |  "
                  f"pass/val: {stats['pass_val']}  |  "
                  f"fail: {stats['fail']}  |  "
                  f"vlm_ann: {stats['ann_json']}")
            for k in total:
                total[k] += stats.get(k, 0)

    print(f"\n{'='*55}")
    print(f"  Hoàn tất!")
    print(f"  Tổng pass/train : {total['pass_train']}")
    print(f"  Tổng pass/val   : {total['pass_val']}")
    print(f"  Tổng fail       : {total['fail']}")
    print(f"  VLM annotations : {total['ann_json']}")
    print(f"{'='*55}")
    print(f"\nBước tiếp theo:")
    print(f"  Train tsfabric_T1:  python -m stage1_anomaly.train --category tsfabric_T1 --eval")
    print(f"  Train tsfabric_T2:  python -m stage1_anomaly.train --category tsfabric_T2 --eval")
    print(f"  Train carpet (MVTec): python -m stage1_anomaly.train --category carpet --eval")


if __name__ == "__main__":
    main()
