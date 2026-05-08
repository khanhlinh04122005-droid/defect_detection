"""
scripts/split_tsfabrics.py

Chia dataset TSfabrics theo SEQUENCE (cuộn vải) — tránh data leakage.

Nguyên tắc:
  - KHÔNG chia theo frame (frame liên tiếp quá giống nhau → data leakage)
  - CHIA theo sequence: mỗi cuộn vải hoặc train HOẶC test, không lẫn lộn

Kết quả:
  data/pass/train/tsfabric_T1/  → Các frame PASS từ cuộn vải TRAIN
  data/pass/test/tsfabric_T1/   → Các frame PASS từ cuộn vải TEST (tính threshold)
  data/fail/train/tsfabric_T1/  → Các frame FAIL từ cuộn vải TRAIN (SAM2, VLM)
  data/fail/test/tsfabric_T1/   → Các frame FAIL từ cuộn vải TEST (AUROC eval)
  data/vlm_ann/                 → JSON annotations tự động sinh

Phân chia 22 sequences:
  T1 (13 sequences): 10 train / 3 test
  T2 (6 sequences) : 4 train  / 2 test
  T3 (3 sequences) : 2 train  / 1 test

Cách chạy:
  python scripts/split_tsfabrics.py
  python scripts/split_tsfabrics.py --test_ratio 0.2 --sample_stride 5
  python scripts/split_tsfabrics.py --dry_run   ← Xem phân chia, không copy file
"""

import argparse
import csv
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Label mapping ─────────────────────────────────────────────────────
PASS_LABELS = {1}
FAIL_LABELS = {2, 4, 5}

DEFECT_TYPE_MAP = {2: "thread", 4: "weave", 5: "hole"}
SEVERITY_MAP    = {2: "minor",  4: "major", 5: "critical"}


def get_fabric_type(seq_name: str) -> str:
    """T1_S148_I108_1 → T1"""
    return seq_name.split("_")[0]


def load_labels(csv_file: Path) -> list[tuple[str, int]]:
    """Đọc CSV → list of (filename, label)"""
    rows = []
    with open(csv_file, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append((row["FileName"], int(row["Label"])))
    return rows


def copy_frames(
    frames: list[tuple[str, int]],
    image_dir: Path,
    dst_dir: Path,
    ann_dir: Path | None,
    vlm_ann_dir: Path,
    seq_name: str,
    stride: int = 1,
    is_fail: bool = False,
) -> dict:
    """
    Copy các frame từ image_dir sang dst_dir.
    stride: lấy 1 frame mỗi N frames (chỉ áp dụng cho pass).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    vlm_ann_dir.mkdir(parents=True, exist_ok=True)

    stats = {"copied": 0, "ann_json": 0}
    fabric_type = get_fabric_type(seq_name)

    for i, (fn, label) in enumerate(frames):
        # Stride sampling chỉ cho ảnh pass (fail giữ tất cả)
        if not is_fail and i % stride != 0:
            continue

        # Tìm file ảnh (CSV ghi .png, file thực là .jpeg)
        src = image_dir / fn.replace(".png", ".jpeg")
        if not src.exists():
            src = image_dir / fn
        if not src.exists():
            continue

        shutil.copy2(src, dst_dir / src.name)
        stats["copied"] += 1

        # Với ảnh fail: tạo VLM annotation JSON
        if is_fail:
            stem = Path(fn).stem
            mask_path = None
            if ann_dir and ann_dir.exists():
                m = ann_dir / f"{stem}.png"
                if m.exists():
                    mask_dst = dst_dir.parent / "masks" / dst_dir.name
                    mask_dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(m, mask_dst / f"{stem}.png")
                    mask_path = str(mask_dst / f"{stem}.png")

            ann = {
                "image":       str(dst_dir / src.name),
                "mask":        mask_path,
                "caption": (
                    f"Lỗi vải loại {DEFECT_TYPE_MAP.get(label, 'other')} "
                    f"được phát hiện trên cuộn vải {fabric_type}. "
                    f"Mức độ nghiêm trọng: {SEVERITY_MAP.get(label, 'major')}."
                ),
                "defect_type": DEFECT_TYPE_MAP.get(label, "other"),
                "severity":    SEVERITY_MAP.get(label, "major"),
                "label_id":    label,
                "sequence":    seq_name,
                "source":      "tsfabrics",
            }
            jf = vlm_ann_dir / f"{seq_name}_{stem}.json"
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(ann, f, ensure_ascii=False, indent=2)
            stats["ann_json"] += 1

    return stats


def split_sequences(sequences: list[str], test_ratio: float, seed: int) -> tuple[list, list]:
    """
    Chia sequences thành train/test THEO TỪNG LOẠI VẢI (T1, T2, T3)
    để đảm bảo mỗi loại đều có đại diện trong test set.
    """
    random.seed(seed)
    by_type = defaultdict(list)
    for s in sequences:
        by_type[get_fabric_type(s)].append(s)

    train_seqs, test_seqs = [], []
    for fabric_type, seqs in sorted(by_type.items()):
        seqs_sorted = sorted(seqs)
        n_test = max(1, math.ceil(len(seqs_sorted) * test_ratio))
        # Lấy test từ cuối list (cuộn vải mới nhất) — không random để reproducible
        test_seqs.extend(seqs_sorted[-n_test:])
        train_seqs.extend(seqs_sorted[:-n_test])

    return train_seqs, test_seqs


def main():
    parser = argparse.ArgumentParser(
        description="Chia TSfabrics theo sequence cho pipeline phát hiện lỗi vải"
    )
    parser.add_argument("--tsfabrics_dir", default="data/TSfabrics")
    parser.add_argument("--output_dir",    default="data")
    parser.add_argument("--fabric_type",   default=None,
                        help="Chỉ xử lý T1, T2, hoặc T3 (mặc định: tất cả)")
    parser.add_argument("--test_ratio",    type=float, default=0.2,
                        help="Tỷ lệ sequence dùng để test (mặc định: 0.2)")
    parser.add_argument("--val_ratio",     type=float, default=0.1,
                        help="Tỷ lệ ảnh PASS/train để tính threshold (mặc định: 0.1)")
    parser.add_argument("--sample_stride", type=int,   default=5,
                        help="Lấy 1 frame mỗi N frame liên tiếp cho ảnh pass (mặc định: 5)")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--dry_run",       action="store_true",
                        help="Xem phân chia mà không copy file thực tế")
    args = parser.parse_args()

    tsfabrics_dir = Path(args.tsfabrics_dir)
    output_dir    = Path(args.output_dir)
    image_root    = tsfabrics_dir / "IMAGE"

    sequences = sorted(d.name for d in image_root.iterdir() if d.is_dir())
    if args.fabric_type:
        sequences = [s for s in sequences if s.startswith(args.fabric_type.upper())]

    train_seqs, test_seqs = split_sequences(sequences, args.test_ratio, args.seed)

    # ── In ra phân chia ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TSfabrics — Sequence-Level Data Split")
    print(f"{'='*60}")
    print(f"  Tổng sequences : {len(sequences)}")
    print(f"  Train sequences: {len(train_seqs)}  ({(1-args.test_ratio)*100:.0f}%)")
    print(f"  Test sequences : {len(test_seqs)}   ({args.test_ratio*100:.0f}%)")
    print(f"  Frame stride   : {args.sample_stride} (pass frames)")
    print(f"{'='*60}")

    # In chi tiết từng loại vải
    for fabric_type in ["T1", "T2", "T3"]:
        tr = [s for s in train_seqs if get_fabric_type(s) == fabric_type]
        te = [s for s in test_seqs  if get_fabric_type(s) == fabric_type]
        if tr or te:
            print(f"\n  Vải {fabric_type}:")
            print(f"    TRAIN ({len(tr)}): {', '.join(tr)}")
            print(f"    TEST  ({len(te)}): {', '.join(te)}")

    if args.dry_run:
        print(f"\n  [DRY RUN] Không copy file. Bỏ --dry_run để thực thi.")
        return

    print(f"\n{'='*60}")
    print(f"  Bắt đầu copy files...")
    print(f"{'='*60}\n")

    grand_total = defaultdict(int)

    for split_name, seq_list in [("train", train_seqs), ("test", test_seqs)]:
        for seq in seq_list:
            fabric_type = get_fabric_type(seq)
            category    = f"tsfabric_{fabric_type}"

            image_dir = tsfabrics_dir / "IMAGE"    / seq
            csv_file  = tsfabrics_dir / "CSV"      / f"{seq}.csv"
            ann_dir   = tsfabrics_dir / "ANNOTATION" / seq / "defect"
            vlm_dir   = output_dir    / "vlm_ann"

            rows = load_labels(csv_file)
            pass_rows = [(f, l) for f, l in rows if l in PASS_LABELS]
            fail_rows = [(f, l) for f, l in rows if l in FAIL_LABELS]

            print(f"[{split_name.upper()}] {seq} → {category}")
            print(f"  pass: {len(pass_rows)} frames | fail: {len(fail_rows)} frames")

            if split_name == "train":
                # ── Pass: chia thêm train/val (để tính threshold) ────
                n_val        = int(len(pass_rows) * args.val_ratio)
                pass_train   = pass_rows[:-n_val] if n_val else pass_rows
                pass_val     = pass_rows[-n_val:] if n_val else []

                dst_pass_train = output_dir / "pass" / "train" / category
                dst_pass_val   = output_dir / "pass" / "test"  / category
                dst_fail       = output_dir / "fail" / "train" / category

                st = copy_frames(pass_train, image_dir, dst_pass_train,
                                 None, vlm_dir, seq, stride=args.sample_stride)
                sv = copy_frames(pass_val,   image_dir, dst_pass_val,
                                 None, vlm_dir, seq, stride=args.sample_stride)
                sf = copy_frames(fail_rows,  image_dir, dst_fail,
                                 ann_dir, vlm_dir, seq, stride=1, is_fail=True)

                print(f"  → pass/train: {st['copied']} | pass/val: {sv['copied']}"
                      f" | fail/train: {sf['copied']} | vlm_ann: {sf['ann_json']}")
                grand_total["pass_train"] += st["copied"]
                grand_total["pass_val"]   += sv["copied"]
                grand_total["fail_train"] += sf["copied"]
                grand_total["vlm_ann"]    += sf["ann_json"]

            else:  # test
                dst_pass_test = output_dir / "pass" / "eval" / category
                dst_fail_test = output_dir / "fail" / "test" / category

                sp = copy_frames(pass_rows, image_dir, dst_pass_test,
                                 None, vlm_dir, seq, stride=args.sample_stride)
                sf = copy_frames(fail_rows, image_dir, dst_fail_test,
                                 ann_dir, vlm_dir, seq, stride=1, is_fail=True)

                print(f"  → pass/eval: {sp['copied']} | fail/test: {sf['copied']}")
                grand_total["pass_eval"]  += sp["copied"]
                grand_total["fail_test"]  += sf["copied"]

    print(f"\n{'='*60}")
    print(f"  ✅ Hoàn tất!")
    print(f"  pass/train (Memory Bank)  : {grand_total['pass_train']:,}")
    print(f"  pass/val  (Threshold)     : {grand_total['pass_val']:,}")
    print(f"  pass/eval (AUROC Good)    : {grand_total['pass_eval']:,}")
    print(f"  fail/train (SAM2 + VLM)  : {grand_total['fail_train']:,}")
    print(f"  fail/test  (AUROC Defect) : {grand_total['fail_test']:,}")
    print(f"  VLM annotations (JSON)    : {grand_total['vlm_ann']:,}")
    print(f"{'='*60}")
    print(f"\nBước tiếp theo:")
    print(f"  python -m stage1_anomaly.train --category tsfabric_T1 --eval")
    print(f"  python -m stage1_anomaly.train --category tsfabric_T2 --eval")
    print(f"  python -m stage1_anomaly.train --category tsfabric_T3 --eval")


if __name__ == "__main__":
    main()
