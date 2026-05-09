"""
stage3_vlm/inference_ollama.py

Stage 3 inference dùng Ollama (chạy CPU — dành cho máy không có GPU).
Thay thế inference.py khi không có VRAM để chạy InternVL2.

Cài đặt:
    1. Download Ollama: https://ollama.com/download
    2. ollama pull moondream   (1.8B, ~1.1GB, nhanh nhất)
       hoặc: ollama pull llava:7b-q4_0   (7B, ~4GB, tốt hơn)
    3. Đảm bảo ollama đang chạy: ollama serve

Dùng:
    from stage3_vlm.inference_ollama import VLMInferenceOllama
    inf = VLMInferenceOllama()
    result = inf.analyze(image_np, mask=mask_np, anomaly_score=0.85)
"""

import base64
import json
import re
from io import BytesIO
from typing import Dict, Optional

import numpy as np
import requests
from PIL import Image


OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL   = "moondream"   # đổi thành "llava:7b-q4_0" nếu muốn tốt hơn


def _encode_image(image_np: np.ndarray) -> str:
    """Encode numpy image (H,W,3) → base64 string cho Ollama API."""
    pil = Image.fromarray(image_np.astype(np.uint8))
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _ollama_chat(model: str, prompt: str, image_b64: str, timeout: int = 60) -> str:
    """Gọi Ollama /api/generate với 1 ảnh."""
    payload = {
        "model":  model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 64,
        },
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama không chạy. Mở terminal và chạy: ollama serve"
        )
    except Exception as e:
        return f"error: {e}"


VQA_PROMPTS = {
    "has_defect": (
        "Look at this fabric image. Does it have any defect or damage? "
        "Reply with only one word: Yes or No."
    ),
    "defect_type": (
        "What type of fabric defect is shown? "
        "Reply with only one word from: hole / tear / stain / yarn / weave / "
        "pilling / discoloration / contamination / other."
    ),
    "severity": (
        "How severe is the fabric defect? "
        "Reply with only one word: minor / major / critical."
    ),
    "location": (
        "Where is the defect located on the fabric? "
        "Reply with a short phrase like: top-left / center / bottom-right / edge."
    ),
    "pass_fail": (
        "Does this fabric meet quality standards for shipping? "
        "Reply with only one word: Pass or Fail."
    ),
}

CAPTION_PROMPT = (
    "You are a fabric quality inspector. "
    "Describe the defect in this fabric image in 1-2 sentences. "
    "Include defect type, location, and severity."
)


class VLMInferenceOllama:
    """
    Stage 3 inference dùng Ollama — chạy CPU, không cần GPU.
    Interface giống VLMInference để dùng thay thế trực tiếp.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model    = model
        self.base_url = base_url
        self._check_connection()

    def _check_connection(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            available = [m for m in models if self.model.split(":")[0] in m]
            if not available:
                print(
                    f"[Ollama] Warning: model '{self.model}' chưa pull.\n"
                    f"  Chạy: ollama pull {self.model}\n"
                    f"  Các model hiện có: {models}"
                )
            else:
                print(f"[Ollama] Connected — model: {available[0]}")
        except Exception:
            print(
                "[Ollama] Warning: không kết nối được. "
                "Đảm bảo 'ollama serve' đang chạy."
            )

    def overlay_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        color=(255, 80, 80),
        alpha: float = 0.4,
    ) -> np.ndarray:
        """Vẽ mask lên ảnh gốc (giống VLMInference.overlay_mask)."""
        import cv2
        overlay     = image.copy().astype(np.float32)
        color_layer = np.zeros_like(image, dtype=np.float32)
        color_layer[mask] = color
        overlay = overlay * (1 - alpha) + color_layer * alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        mask_u8   = mask.astype(np.uint8) * 255
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
        Phân tích lỗi từ 1 ảnh — interface giống VLMInference.analyze().

        Args:
            image:         (H, W, 3) uint8 RGB
            mask:          (H, W) bool — từ Stage 2 (None nếu không có)
            anomaly_score: float — từ Stage 1

        Returns:
            dict: caption, defect_type, severity, location, pass_fail,
                  has_defect, confidence, vqa_raw
        """
        viz = image
        if mask is not None and mask.any():
            viz = self.overlay_mask(image, mask)

        img_b64 = _encode_image(viz)

        # Caption
        caption = _ollama_chat(
            self.model,
            CAPTION_PROMPT + f"\nAnomaly score: {anomaly_score:.3f}",
            img_b64,
            timeout=90,
        )

        # VQA
        answers = {}
        for key, prompt in VQA_PROMPTS.items():
            answers[key] = _ollama_chat(self.model, prompt, img_b64, timeout=60)

        # Parse pass_fail
        pf_raw    = answers.get("pass_fail", "").lower()
        pass_fail = "Fail" if "fail" in pf_raw else "Pass"

        # Parse has_defect
        hd_raw    = answers.get("has_defect", "").lower()
        has_defect = "yes" in hd_raw or "có" in hd_raw

        confidence = min(1.0, anomaly_score / 10.0 + 0.5) if has_defect else 0.1

        return {
            "caption":     caption,
            "defect_type": _extract_word(answers.get("defect_type", ""), [
                "hole", "tear", "stain", "yarn", "weave",
                "pilling", "discoloration", "contamination", "other",
            ]),
            "severity":    _extract_word(answers.get("severity", ""), ["minor", "major", "critical"]),
            "location":    answers.get("location", "unknown"),
            "pass_fail":   pass_fail,
            "has_defect":  has_defect,
            "confidence":  round(confidence, 3),
            "vqa_raw":     answers,
        }


def _extract_word(text: str, valid_words: list) -> str:
    """Lấy từ hợp lệ đầu tiên trong chuỗi trả về của model."""
    text_lower = text.lower()
    for w in valid_words:
        if w in text_lower:
            return w
    return text.split()[0].lower() if text.strip() else "unknown"


if __name__ == "__main__":
    import sys
    import numpy as np
    from PIL import Image as PILImage

    img_path = sys.argv[1] if len(sys.argv) > 1 else "data/fail/images/tsfabric_T1/000003.jpeg"
    model    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL

    print(f"=== Ollama Stage 3 Test ===")
    print(f"Image : {img_path}")
    print(f"Model : {model}")

    inf   = VLMInferenceOllama(model=model)
    image = np.array(PILImage.open(img_path).convert("RGB"))

    result = inf.analyze(image, mask=None, anomaly_score=0.85)

    print(f"\nCaption    : {result['caption']}")
    print(f"Defect type: {result['defect_type']}")
    print(f"Severity   : {result['severity']}")
    print(f"Location   : {result['location']}")
    print(f"Pass/Fail  : {result['pass_fail']}")
    print(f"Has defect : {result['has_defect']}")
    print(f"Confidence : {result['confidence']}")
