import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

from configs.base_config import DEVICE, CATEGORY
from configs.stage3_config import VQA_QUESTIONS, LORA_WEIGHTS_PATH
from stage3_vlm.model import InternVL2Wrapper


class VLMInference:
    """
    Inference wrapper cho Stage 3 — dùng sau khi đã fine-tune.
    Nhận ảnh + mask overlay từ Stage 2, trả về caption + VQA answers.
    """

    def __init__(
        self,
        device: str = DEVICE,
        lora_weights: str = None,
    ):
        self.device = device
        self.model  = InternVL2Wrapper(
            device       = device,
            use_lora     = True,
            lora_weights = lora_weights or LORA_WEIGHTS_PATH,
        )
        self.model.eval_mode()

    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color=(255, 80, 80),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Vẽ mask lên ảnh gốc với màu bán trong suốt.

        Args:
            image: (H, W, 3) uint8 RGB
            mask:  (H, W) bool
            color: màu highlight vùng lỗi
            alpha: độ trong suốt overlay

        Returns:
            ảnh overlay (H, W, 3) uint8
        """
        import cv2
        overlay = image.copy().astype(np.float32)
        color_layer = np.zeros_like(image, dtype=np.float32)
        color_layer[mask] = color

        overlay = overlay * (1 - alpha) + color_layer * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Vẽ contour quanh vùng lỗi
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        return overlay

    def analyze(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        anomaly_score: float = 0.0,
    ) -> Dict:
        """
        Phân tích lỗi từ 1 ảnh.

        Args:
            image:         (H, W, 3) uint8 RGB — ảnh gốc
            mask:          (H, W) bool — mask từ Stage 2 (None nếu chưa có)
            anomaly_score: float — từ Stage 1

        Returns:
            dict với keys:
                caption, defect_type, severity, location, pass_fail,
                has_defect, confidence
        """
        # Nếu có mask: overlay lên ảnh để VLM thấy vùng lỗi rõ hơn
        viz_image = image
        if mask is not None and mask.any():
            viz_image = self.overlay_mask(image, mask)

        # Sinh caption
        extra = f"Anomaly score: {anomaly_score:.3f}"
        caption = self.model.caption(viz_image, extra_context=extra)

        # VQA
        answers = self.model.vqa(viz_image)

        # Parse pass_fail từ VQA answer
        pf_raw = answers.get("pass_fail", "").lower()
        pass_fail = "Fail" if "fail" in pf_raw else "Pass"

        # Parse has_defect
        has_defect_raw = answers.get("has_defect", "").lower()
        has_defect = "có" in has_defect_raw or "yes" in has_defect_raw

        # Confidence đơn giản: dựa trên anomaly_score + VQA consistency
        confidence = min(1.0, anomaly_score / 10.0 + 0.5) if has_defect else 0.1

        return {
            "caption":     caption,
            "defect_type": answers.get("defect_type", "unknown"),
            "severity":    answers.get("severity",    "unknown"),
            "location":    answers.get("location",    "unknown"),
            "pass_fail":   pass_fail,
            "has_defect":  has_defect,
            "confidence":  round(confidence, 3),
            "vqa_raw":     answers,
        }


if __name__ == "__main__":
    import numpy as np
    print("=== VLMInference smoke test ===")
    print("(Không load model thật — chỉ test overlay)")

    inf_obj = object.__new__(VLMInference)
    inf_obj.device = "cpu"

    dummy_img  = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    dummy_mask = np.zeros((224, 224), dtype=bool)
    dummy_mask[80:140, 80:140] = True

    overlaid = inf_obj.overlay_mask(dummy_img, dummy_mask)
    print(f"Overlay output shape: {overlaid.shape}")
    print("OK")
