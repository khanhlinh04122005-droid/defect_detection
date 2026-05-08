"""
data_pipeline/synthetic_gen.py

Sinh dữ liệu lỗi giả tổng hợp khi thiếu ảnh fail thực.

Phương pháp:
    1. CutPaste   — cắt dán patch từ chính ảnh pass
    2. Texture    — chèn texture lỗi (scratch, stain) lên ảnh pass
    3. Noise Spot — tạo vùng nhiễu cục bộ

Cách dùng:
    python data_pipeline/synthetic_gen.py --category metal_nut --n 100
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from configs.base_config import PASS_TRAIN_DIR, FAIL_IMAGES_DIR, FAIL_MASKS_DIR, CATEGORY
from data_pipeline.cutpaste import CutPaste


class SyntheticDefectGenerator:
    """
    Sinh ảnh lỗi giả từ ảnh pass sử dụng nhiều phương pháp.
    """

    def __init__(self, seed: int = 42):
        self.rng       = np.random.default_rng(seed)
        self.cutpaste  = CutPaste()
        random.seed(seed)

    def generate(
        self,
        image: np.ndarray,
        method: str = "random",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sinh 1 ảnh lỗi giả từ ảnh pass.

        Args:
            image:  (H, W, 3) uint8 RGB
            method: "cutpaste" | "scar" | "scratch" | "stain" | "noise_spot" | "random"

        Returns:
            (defect_image, mask) — mask (H, W) uint8 binary
        """
        methods = ["cutpaste", "scar", "scratch", "stain", "noise_spot"]
        if method == "random":
            method = random.choice(methods)

        if method in ("cutpaste", "scar"):
            return self.cutpaste(image, variant=method)
        elif method == "scratch":
            return self._gen_scratch(image)
        elif method == "stain":
            return self._gen_stain(image)
        elif method == "noise_spot":
            return self._gen_noise_spot(image)
        else:
            return self.cutpaste(image, "cutpaste")

    def _gen_scratch(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo vết xước thẳng với màu tối hơn."""
        h, w  = image.shape[:2]
        result = image.copy()
        mask   = np.zeros((h, w), dtype=np.uint8)

        # Tham số vết xước
        x0 = random.randint(0, w)
        y0 = random.randint(0, h)
        length = random.randint(int(min(h, w) * 0.1), int(min(h, w) * 0.5))
        angle  = random.uniform(0, np.pi)
        width  = random.randint(1, 5)

        x1 = int(x0 + length * np.cos(angle))
        y1 = int(y0 + length * np.sin(angle))

        # Màu vết xước: tối hơn pixel xung quanh
        region = image[max(0,y0-5):min(h,y0+5), max(0,x0-5):min(w,x0+5)]
        base_color = region.mean(axis=(0,1)) if region.size > 0 else np.array([128,128,128])
        scratch_color = np.clip(base_color * 0.4, 0, 255).astype(int).tolist()

        cv2.line(result, (x0, y0), (x1, y1), scratch_color, width)
        cv2.line(mask,   (x0, y0), (x1, y1), 255, width + 2)

        return result, mask

    def _gen_stain(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo vết bẩn dạng blob ngẫu nhiên."""
        h, w  = image.shape[:2]
        result = image.copy().astype(np.float32)
        mask   = np.zeros((h, w), dtype=np.uint8)

        # Tâm và bán kính blob
        cx = random.randint(w // 5, 4 * w // 5)
        cy = random.randint(h // 5, 4 * h // 5)
        rx = random.randint(int(w * 0.04), int(w * 0.15))
        ry = random.randint(int(h * 0.04), int(h * 0.15))

        # Màu stain: vàng nâu hoặc đen
        stain_type = random.choice(["dark", "rust", "white"])
        colors = {
            "dark":  [20,  15,  10 ],
            "rust":  [180, 80,  20 ],
            "white": [240, 240, 235],
        }
        stain_color = np.array(colors[stain_type], dtype=np.float32)

        # Tạo ellipse mask với alpha
        ell_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(ell_mask, (cx, cy), (rx, ry),
                    random.uniform(0, 180), 0, 360, 1.0, -1)

        # Gaussian blur để mép mềm hơn
        ell_mask = cv2.GaussianBlur(ell_mask, (21, 21), 0)

        alpha = ell_mask[:, :, None] * random.uniform(0.4, 0.85)
        result = result * (1 - alpha) + stain_color * alpha
        result = np.clip(result, 0, 255).astype(np.uint8)

        mask[ell_mask > 0.1] = 255

        return result, mask

    def _gen_noise_spot(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tạo vùng nhiễu cục bộ (texture anomaly)."""
        h, w  = image.shape[:2]
        result = image.copy()
        mask   = np.zeros((h, w), dtype=np.uint8)

        # Patch nhiễu
        ph = random.randint(int(h * 0.05), int(h * 0.2))
        pw = random.randint(int(w * 0.05), int(w * 0.2))
        py = random.randint(0, h - ph)
        px = random.randint(0, w - pw)

        noise_type = random.choice(["gaussian", "salt_pepper", "uniform"])

        patch = image[py:py+ph, px:px+pw].astype(np.float32)

        if noise_type == "gaussian":
            noise = np.random.normal(0, random.uniform(20, 60), patch.shape)
        elif noise_type == "salt_pepper":
            noise = np.zeros(patch.shape)
            sp_mask = np.random.rand(*patch.shape[:2])
            noise[sp_mask < 0.05]  = -128
            noise[sp_mask > 0.95]  =  128
        else:
            noise = np.random.uniform(-40, 40, patch.shape)

        patched = np.clip(patch + noise, 0, 255).astype(np.uint8)
        result[py:py+ph, px:px+pw] = patched
        mask[py:py+ph, px:px+pw]   = 255

        return result, mask


def generate(
    category:    str,
    n:           int           = 100,
    methods:     List[str]     = None,
    pass_dir:    str           = None,
    out_img_dir: str           = None,
    out_mask_dir: str          = None,
    seed:        int           = 42,
):
    """
    Sinh n ảnh lỗi giả cho category.

    Args:
        category:  tên category
        n:         số ảnh muốn sinh
        methods:   list phương pháp muốn dùng (None = random)
    """
    pass_path = Path(pass_dir or PASS_TRAIN_DIR)
    out_img   = Path(out_img_dir  or FAIL_IMAGES_DIR) / category
    out_mask  = Path(out_mask_dir or FAIL_MASKS_DIR)  / category
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    pass_imgs = list(pass_path.glob("*.png")) + list(pass_path.glob("*.jpg"))
    if not pass_imgs:
        print(f"[SyntheticGen] Không tìm thấy ảnh pass trong {pass_path}")
        return 0

    generator = SyntheticDefectGenerator(seed=seed)
    all_methods = methods or ["cutpaste", "scar", "scratch", "stain", "noise_spot"]
    generated = 0

    for i in range(n):
        img_path = random.choice(pass_imgs)
        img_np   = np.array(Image.open(img_path).convert("RGB"))
        method   = all_methods[i % len(all_methods)]

        aug, mask = generator.generate(img_np, method=method)

        fname = f"synthetic_{category}_{method}_{i:04d}.png"
        Image.fromarray(aug).save(out_img / fname)
        Image.fromarray(mask).save(out_mask / fname)
        generated += 1

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n}] Generated...")

    print(f"[SyntheticGen] Done: {generated} images → {out_img}")
    return generated


def main():
    parser = argparse.ArgumentParser(description="Synthetic defect data generation")
    parser.add_argument("--category",  default=CATEGORY)
    parser.add_argument("--n",         type=int, default=100, help="Số ảnh cần sinh")
    parser.add_argument("--methods",   nargs="+",
                        choices=["cutpaste", "scar", "scratch", "stain", "noise_spot"],
                        default=None, help="Phương pháp sinh (mặc định: random)")
    parser.add_argument("--pass_dir",  default=None)
    parser.add_argument("--out_img",   default=None)
    parser.add_argument("--out_mask",  default=None)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    print(f"[SyntheticGen] Category: {args.category} | N: {args.n}")
    generate(
        category    = args.category,
        n           = args.n,
        methods     = args.methods,
        pass_dir    = args.pass_dir,
        out_img_dir = args.out_img,
        out_mask_dir= args.out_mask,
        seed        = args.seed,
    )


if __name__ == "__main__":
    main()
