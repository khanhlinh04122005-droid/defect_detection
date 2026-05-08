"""
data_pipeline/cutpaste.py

CutPaste augmentation: tạo ảnh lỗi giả bằng cách cắt một patch
từ ảnh pass rồi dán vào vị trí ngẫu nhiên.

Paper: CutPaste: Self-Supervised Learning for Anomaly Detection
       and Localization (Li et al., CVPR 2021)

Dùng để tạo thêm training data cho Stage 1 và Stage 2.
"""

import random
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image


class CutPaste:
    """
    Tạo ảnh lỗi giả bằng cách cắt-dán patch.

    Variants:
        - "cutpaste"    : Cắt patch → dán vào vị trí khác
        - "scar"        : Tạo vết xước thẳng hẹp
    """

    def __init__(
        self,
        patch_ratio:     Tuple[float, float] = (0.02, 0.15),
        aspect_ratio:    Tuple[float, float] = (0.3, 3.3),
        rotation:        Tuple[float, float] = (-180, 180),
        colorjitter:     float = 0.1,
        scar_width:      Tuple[int, int]     = (2, 16),
        scar_length:     Tuple[float, float] = (0.1, 0.25),
    ):
        self.patch_ratio  = patch_ratio
        self.aspect_ratio = aspect_ratio
        self.rotation     = rotation
        self.colorjitter  = colorjitter
        self.scar_width   = scar_width
        self.scar_length  = scar_length

    def cutpaste(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutPaste cơ bản: cắt 1 patch → dán vào vị trí mới.

        Returns:
            (augmented_image, binary_mask) — mask chỉ vùng đã dán
        """
        h, w   = image.shape[:2]
        result = image.copy()
        mask   = np.zeros((h, w), dtype=np.uint8)

        area    = h * w * random.uniform(*self.patch_ratio)
        ar      = random.uniform(*self.aspect_ratio)
        ph      = int(np.sqrt(area * ar))
        pw      = int(np.sqrt(area / ar))
        ph      = max(4, min(ph, h - 1))
        pw      = max(4, min(pw, w - 1))

        # Vị trí cắt
        sy = random.randint(0, h - ph)
        sx = random.randint(0, w - pw)
        patch = image[sy:sy+ph, sx:sx+pw].copy()

        # Color jitter nhẹ cho patch
        if self.colorjitter > 0:
            noise = np.random.uniform(
                -255 * self.colorjitter,
                 255 * self.colorjitter,
                patch.shape
            ).astype(np.float32)
            patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Rotation
        angle = random.uniform(*self.rotation)
        M     = cv2.getRotationMatrix2D((pw/2, ph/2), angle, 1.0)
        patch = cv2.warpAffine(patch, M, (pw, ph),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

        # Vị trí dán
        dy = random.randint(0, h - ph)
        dx = random.randint(0, w - pw)

        result[dy:dy+ph, dx:dx+pw] = patch
        mask[dy:dy+ph, dx:dx+pw]   = 255

        return result, mask

    def scar(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        CutPaste-Scar: tạo vết xước thẳng hẹp.

        Returns:
            (augmented_image, binary_mask)
        """
        h, w   = image.shape[:2]
        result = image.copy()
        mask   = np.zeros((h, w), dtype=np.uint8)

        sw      = random.randint(*self.scar_width)
        sl_h    = int(h * random.uniform(*self.scar_length))
        sl_w    = int(w * random.uniform(*self.scar_length))

        # Cắt scar patch (dọc hoặc ngang)
        if random.random() > 0.5:
            ph, pw = sl_h, sw
        else:
            ph, pw = sw, sl_w

        ph = max(2, min(ph, h - 1))
        pw = max(2, min(pw, w - 1))

        sy = random.randint(0, h - ph)
        sx = random.randint(0, w - pw)
        patch = image[sy:sy+ph, sx:sx+pw].copy()

        # Color jitter
        noise = np.random.uniform(-30, 30, patch.shape).astype(np.float32)
        patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Dán vào vị trí mới
        dy = random.randint(0, h - ph)
        dx = random.randint(0, w - pw)
        result[dy:dy+ph, dx:dx+pw] = patch
        mask[dy:dy+ph, dx:dx+pw]   = 255

        return result, mask

    def __call__(
        self,
        image: np.ndarray,
        variant: str = "cutpaste",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            image:   (H, W, 3) uint8 RGB — ảnh pass gốc
            variant: "cutpaste" | "scar" | "random"

        Returns:
            (augmented, mask) — mask là binary map vùng lỗi giả
        """
        if variant == "random":
            variant = random.choice(["cutpaste", "scar"])

        if variant == "scar":
            return self.scar(image)
        return self.cutpaste(image)


def generate_synthetic_dataset(
    pass_dir:  str,
    output_img_dir: str,
    output_mask_dir: str,
    n_per_image: int = 3,
    variants: List[str] = ("cutpaste", "scar"),
    seed: int = 42,
):
    """
    Sinh dataset lỗi giả từ ảnh pass.

    Args:
        pass_dir:        thư mục chứa ảnh pass
        output_img_dir:  thư mục lưu ảnh lỗi giả
        output_mask_dir: thư mục lưu mask tương ứng
        n_per_image:     số ảnh lỗi sinh ra từ mỗi ảnh pass
        variants:        các loại CutPaste dùng
    """
    random.seed(seed)
    np.random.seed(seed)

    pass_path = Path(pass_dir)
    out_img   = Path(output_img_dir)
    out_mask  = Path(output_mask_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_mask.mkdir(parents=True, exist_ok=True)

    augmentor = CutPaste()
    images    = list(pass_path.glob("*.png")) + list(pass_path.glob("*.jpg"))

    total = 0
    for img_path in images:
        img_np = np.array(Image.open(img_path).convert("RGB"))

        for i in range(n_per_image):
            variant = variants[i % len(variants)]
            aug, mask = augmentor(img_np, variant=variant)

            stem  = img_path.stem
            fname = f"{stem}_cutpaste_{variant}_{i:02d}.png"

            Image.fromarray(aug).save(out_img / fname)
            Image.fromarray(mask).save(out_mask / fname)
            total += 1

    print(f"[CutPaste] Generated {total} synthetic images → {out_img}")
    return total


if __name__ == "__main__":
    print("=== CutPaste smoke test ===")
    augmentor = CutPaste()
    dummy     = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)

    aug_cp, mask_cp = augmentor(dummy, "cutpaste")
    aug_sc, mask_sc = augmentor(dummy, "scar")

    print(f"CutPaste — aug: {aug_cp.shape}, mask coverage: {mask_cp.mean():.1f}")
    print(f"Scar     — aug: {aug_sc.shape}, mask coverage: {mask_sc.mean():.1f}")
    print("OK")
