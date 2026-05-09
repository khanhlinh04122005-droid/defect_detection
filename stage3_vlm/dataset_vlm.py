import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from configs.base_config import VLM_ANN_DIR
from configs.stage3_config import (
    CAPTION_MAX_LENGTH, SYSTEM_PROMPT_CAPTION, ANNOTATION_FORMAT
)


class VLMDataset(Dataset):
    """
    Dataset cho fine-tune InternVL2 — Tương thích TSfabrics.

    Nguồn dữ liệu: data/vlm_ann/*.json  (tự sinh bởi split_tsfabrics.py)

    Format JSON annotation (TSfabrics):
        {
          "image":       "data/fail/train/tsfabric_T1/000084.jpeg",
          "mask":        "data/fail/masks/tsfabric_T1/000084.png",
          "caption":     "Lỗi vải loại weave trên cuộn vải T1. Mức độ: major.",
          "defect_type": "weave",
          "severity":    "major",
          "source":      "tsfabrics",
          "sequence":    "T1_S148_I108_1",
          "label_id":    4
        }
    """

    IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]

    def __init__(
        self,
        ann_dir: str = None,
        tokenizer=None,
        max_length: int = CAPTION_MAX_LENGTH,
        augment: bool = False,
    ):
        self.ann_dir   = Path(ann_dir or VLM_ANN_DIR)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment   = augment

        self.samples: List[Tuple[Path, Dict]] = []
        self._load_samples()

        print(f"[VLMDataset] {len(self.samples)} samples ← {self.ann_dir}")

    def _load_samples(self):
        if not self.ann_dir.exists():
            print(f"[VLMDataset] Warning: {self.ann_dir} không tồn tại")
            return

        for json_path in sorted(self.ann_dir.glob("*.json")):
            with open(json_path, encoding="utf-8") as f:
                ann = json.load(f)

            if "caption" not in ann:
                print(f"[VLMDataset] Skip {json_path.name}: thiếu 'caption'")
                continue

            # Ư u tiên lấy đường dẫn ảnh từ field 'image' (TSfabrics JSON)
            img_path = None
            if "image" in ann:
                candidate = Path(ann["image"])
                if candidate.exists():
                    img_path = candidate

            # Fallback: tìm file ảnh cùng tên với JSON
            if img_path is None:
                for ext in self.IMG_EXTS:
                    candidate = json_path.with_suffix(ext)
                    if candidate.exists():
                        img_path = candidate
                        break

            if img_path is None:
                continue

            self.samples.append((img_path, ann))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ann = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))   # (H, W, 3) uint8

        if self.augment:
            image = self._augment(image)

        caption     = ann.get("caption", "")
        defect_type = ann.get("defect_type", "unknown")
        severity    = ann.get("severity",    "unknown")
        location    = ann.get("location",    "unknown")

        # Prompt đầy đủ để train
        prompt = (
            f"{SYSTEM_PROMPT_CAPTION}\n"
            f"<image>\nMô tả lỗi:"
        )
        target = caption

        item = {
            "image":       image,
            "prompt":      prompt,
            "target":      target,
            "defect_type": defect_type,
            "severity":    severity,
            "location":    location,
            "image_path":  str(img_path),
        }

        # Tokenize nếu có tokenizer
        if self.tokenizer is not None:
            # Tokenize full_text = prompt + target để model học predict caption
            full_text = prompt + " " + target

            enc = self.tokenizer(
                full_text,
                truncation     = True,
                max_length     = self.max_length,
                padding        = "max_length",
                return_tensors = "pt",
            )
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)

            # Mask phần prompt trong labels bằng -100 (không tính loss)
            prompt_enc = self.tokenizer(
                prompt,
                truncation     = True,
                max_length     = self.max_length,
                return_tensors = "pt",
            )
            prompt_len = prompt_enc["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:prompt_len] = -100   # bỏ qua loss trên prompt

            item["input_ids"]      = input_ids
            item["attention_mask"] = attention_mask
            item["labels"]         = labels

        return item

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Augmentation nhẹ cho ảnh: flip ngang ngẫu nhiên."""
        if np.random.rand() > 0.5:
            image = image[:, ::-1].copy()
        return image

    def get_annotations(self) -> List[Dict]:
        """Trả về list tất cả annotation dict."""
        return [ann for _, ann in self.samples]

    def class_distribution(self) -> Dict[str, int]:
        """Thống kê phân bố defect_type."""
        from collections import Counter
        types = [ann.get("defect_type", "unknown") for _, ann in self.samples]
        return dict(Counter(types))


def collate_vlm(batch: List[Dict]) -> Dict:
    """
    Collate function cho VLMDataset.
    Giữ images dưới dạng list (kích thước có thể khác nhau).
    """
    keys = batch[0].keys()
    out  = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out


if __name__ == "__main__":
    print("=== VLMDataset smoke test ===")
    ds = VLMDataset()
    print(f"Samples: {len(ds)}")
    if len(ds) > 0:
        item = ds[0]
        print(f"Image shape  : {item['image'].shape}")
        print(f"Caption      : {item['target'][:80]}")
        print(f"Defect type  : {item['defect_type']}")
        print(f"Severity     : {item['severity']}")
    print("OK")
