"""
data_pipeline/augmentation.py

Augmentation nâng cao cho ảnh công nghiệp:
    - Standard augmentation (flip, rotate, color jitter, v.v.)
    - Industrial augmentation (brightness gradient, blur, noise)
    - Dùng ở bất kỳ stage nào cần augment data
"""

import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
from typing import Tuple, Optional


class IndustrialAugment:
    """
    Augmentation pipeline chuyên cho ảnh công nghiệp.
    Tất cả methods nhận và trả về numpy (H,W,3) uint8 RGB.
    """

    def __init__(
        self,
        flip_p:        float = 0.5,
        rotate_deg:    float = 15.0,
        scale_range:   Tuple = (0.85, 1.15),
        brightness:    float = 0.2,
        contrast:      float = 0.2,
        saturation:    float = 0.1,
        blur_p:        float = 0.2,
        noise_p:       float = 0.2,
        noise_std:     float = 8.0,
        jpeg_p:        float = 0.1,
        jpeg_quality:  Tuple = (60, 95),
        seed:          Optional[int] = None,
    ):
        self.flip_p       = flip_p
        self.rotate_deg   = rotate_deg
        self.scale_range  = scale_range
        self.brightness   = brightness
        self.contrast     = contrast
        self.saturation   = saturation
        self.blur_p       = blur_p
        self.noise_p      = noise_p
        self.noise_std    = noise_std
        self.jpeg_p       = jpeg_p
        self.jpeg_quality = jpeg_quality

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Áp dụng toàn bộ pipeline augmentation."""
        img = image.copy()
        img = self.random_flip(img)
        img = self.random_rotate(img)
        img = self.random_scale_crop(img)
        img = self.color_jitter(img)
        img = self.random_blur(img)
        img = self.random_noise(img)
        img = self.random_jpeg(img)
        return img

    def random_flip(self, img: np.ndarray) -> np.ndarray:
        if random.random() < self.flip_p:
            img = img[:, ::-1].copy()           # Flip ngang
        if random.random() < self.flip_p * 0.3:
            img = img[::-1, :].copy()           # Flip dọc (ít hơn)
        return img

    def random_rotate(self, img: np.ndarray) -> np.ndarray:
        if self.rotate_deg <= 0:
            return img
        angle  = random.uniform(-self.rotate_deg, self.rotate_deg)
        h, w   = img.shape[:2]
        M      = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT_101)

    def random_scale_crop(self, img: np.ndarray) -> np.ndarray:
        h, w  = img.shape[:2]
        scale = random.uniform(*self.scale_range)
        nh, nw = int(h * scale), int(w * scale)

        # Resize
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Center crop về kích thước gốc
        if nh >= h and nw >= w:
            y0 = (nh - h) // 2
            x0 = (nw - w) // 2
            return resized[y0:y0+h, x0:x0+w]
        else:
            # Pad
            pad_h = max(0, h - nh)
            pad_w = max(0, w - nw)
            return cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_REFLECT_101)

    def color_jitter(self, img: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(img)

        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            pil = ImageEnhance.Brightness(pil).enhance(factor)

        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            pil = ImageEnhance.Contrast(pil).enhance(factor)

        if self.saturation > 0:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            pil = ImageEnhance.Color(pil).enhance(factor)

        return np.array(pil)

    def random_blur(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.blur_p:
            return img
        k = random.choice([3, 5])   # Kernel size
        return cv2.GaussianBlur(img, (k, k), 0)

    def random_noise(self, img: np.ndarray) -> np.ndarray:
        if random.random() > self.noise_p:
            return img
        noise  = np.random.normal(0, self.noise_std, img.shape).astype(np.float32)
        noisy  = img.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def random_jpeg(self, img: np.ndarray) -> np.ndarray:
        """Giả lập nén JPEG để tăng robustness."""
        if random.random() > self.jpeg_p:
            return img
        quality = random.randint(*self.jpeg_quality)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), encode_param)
        decoded   = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


# Preset nhanh

def get_train_augment() -> IndustrialAugment:
    """Augmentation mạnh cho training."""
    return IndustrialAugment(
        flip_p=0.5, rotate_deg=20.0, scale_range=(0.8, 1.2),
        brightness=0.25, contrast=0.25, saturation=0.15,
        blur_p=0.25, noise_p=0.25, noise_std=10.0, jpeg_p=0.15,
    )


def get_val_augment() -> IndustrialAugment:
    """Augmentation nhẹ cho validation (chỉ flip)."""
    return IndustrialAugment(
        flip_p=0.5, rotate_deg=0, scale_range=(1.0, 1.0),
        brightness=0, contrast=0, saturation=0,
        blur_p=0, noise_p=0, jpeg_p=0,
    )


if __name__ == "__main__":
    import os
    print("=== Augmentation smoke test ===")
    aug = get_train_augment()
    dummy = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    out   = aug(dummy)
    print(f"Input : {dummy.shape} | min={dummy.min()} max={dummy.max()}")
    print(f"Output: {out.shape}   | min={out.min()}   max={out.max()}")
    print("OK")
