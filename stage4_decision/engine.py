"""
stage4_decision/engine.py

Pipeline engine — chạy tuần tự Stage 1 → 2 → 3 → 4 cho 1 ảnh.
Dùng trong inference và batch processing.
"""

import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, Optional

from configs.base_config import DEVICE, CATEGORY
from stage1_anomaly.patchcore  import PatchCore
from stage2_seg.sam2_wrapper   import SAM2Wrapper
from stage3_vlm.inference      import VLMInference
from stage4_decision.rules     import DecisionInput, DecisionOutput, decide


class PipelineEngine:
    """
    Full 4-stage defect detection pipeline.

    Lazy loading: mỗi stage chỉ load khi cần (tiết kiệm VRAM).
    Có thể chạy stage1_only nếu chưa có Stage 2-3.
    """

    def __init__(
        self,
        category:     str  = CATEGORY,
        device:       str  = DEVICE,
        stage1_only:  bool = False,
        stage12_only: bool = False,
    ):
        self.category     = category
        self.device       = device
        self.stage1_only  = stage1_only
        self.stage12_only = stage12_only

        self._stage1: Optional[PatchCore]    = None
        self._stage2: Optional[SAM2Wrapper]  = None
        self._stage3: Optional[VLMInference] = None

    def _get_stage1(self) -> PatchCore:
        if self._stage1 is None:
            self._stage1 = PatchCore(category=self.category, device=self.device)
            self._stage1.load()
        return self._stage1

    def _get_stage2(self) -> SAM2Wrapper:
        if self._stage2 is None:
            self._stage2 = SAM2Wrapper(device=self.device, use_lora=True)
        return self._stage2

    def _get_stage3(self) -> VLMInference:
        if self._stage3 is None:
            self._stage3 = VLMInference(device=self.device)
        return self._stage3

    def run(self, image_path: str) -> Dict:
        """
        Chạy full pipeline cho 1 ảnh.

        Args:
            image_path: đường dẫn ảnh cần kiểm tra

        Returns:
            dict kết quả đầy đủ từ tất cả stages + decision
        """
        t_start = time.time()
        image_path = str(image_path)

        result = {
            "image_path": image_path,
            "category":   self.category,
            "stage1":     {},
            "stage2":     {},
            "stage3":     {},
            "decision":   {},
            "elapsed_ms": 0,
        }

        # Stage 1 — Anomaly Detection
        t1 = time.time()
        s1 = self._get_stage1().predict(image_path)
        result["stage1"] = {
            "score":      s1["image_score"],
            "threshold":  s1["threshold"],
            "prediction": s1["prediction"],
            "score_map":  s1["score_map"],   # (H, W) numpy — không serialize vào JSON
        }
        result["stage1"]["elapsed_ms"] = round((time.time() - t1) * 1000)

        if self.stage1_only or s1["prediction"] == "Pass":
            if s1["prediction"] == "Pass":
                result["decision"] = _pass_decision()
            else:
                result["decision"] = _fail_decision_stage1_only(s1)
            result["stage2"] = {}
            result["stage3"] = {}
            result["elapsed_ms"] = round((time.time() - t_start) * 1000)
            return result

        # Stage 2 — Segmentation
        t2 = time.time()
        img_np    = np.array(Image.open(image_path).convert("RGB"))
        score_map = s1["score_map"]

        mask, sam_conf = self._get_stage2().predict_mask(img_np, score_map)
        mask_area      = float(mask.sum()) / max(mask.size, 1)

        result["stage2"] = {
            "mask":        mask,          # (H, W) bool numpy
            "confidence":  sam_conf,
            "area_ratio":  round(mask_area, 4),
            "elapsed_ms":  round((time.time() - t2) * 1000),
        }

        if self.stage12_only:
            inp = _build_input(self.category, result["stage1"], result["stage2"])
            out = decide(inp)
            result["decision"] = _format_decision(out)
            result["elapsed_ms"] = round((time.time() - t_start) * 1000)
            return result

        # Unload Stage 2 trước khi load Stage 3 — tránh OOM trên 4GB VRAM
        self._stage2 = None
        import gc, torch as _torch
        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

        # Stage 3 — VLM
        t3 = time.time()
        vlm_out = self._get_stage3().analyze(
            image         = img_np,
            mask          = mask,
            anomaly_score = s1["image_score"],
        )
        result["stage3"] = {**vlm_out, "elapsed_ms": round((time.time() - t3) * 1000)}

        # Stage 4 — Decision
        inp = _build_input(self.category, result["stage1"], result["stage2"], result["stage3"])
        out = decide(inp)
        result["decision"] = _format_decision(out)
        result["elapsed_ms"] = round((time.time() - t_start) * 1000)

        return result

    def unload(self):
        """Giải phóng VRAM."""
        self._stage1 = None
        self._stage2 = None
        self._stage3 = None
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _build_input(category, s1, s2=None, s3=None) -> DecisionInput:
    s2 = s2 or {}
    s3 = s3 or {}

    # Tính mask_iou an toàn — mask có thể là numpy array hoặc None
    mask_np = s2.get("mask")
    mask_area = s2.get("area_ratio", 0.0)
    if mask_np is not None and hasattr(mask_np, "sum"):
        total = mask_np.size or 1
        mask_area = float(mask_np.sum()) / total

    return DecisionInput(
        anomaly_score    = float(s1.get("score", 0.0)),
        stage1_threshold = float(s1.get("threshold", 0.5)),
        stage1_pred      = s1.get("prediction", "Pass"),
        mask_iou         = float(s2.get("iou", 0.0)),
        mask_area_ratio  = float(mask_area),
        sam_confidence   = float(s2.get("confidence", 0.0)),
        defect_type      = s3.get("defect_type", "unknown") or "unknown",
        severity         = s3.get("severity",    "unknown") or "unknown",
        location         = s3.get("location",    "unknown") or "unknown",
        vlm_pass_fail    = s3.get("pass_fail",   "Pass")   or "Pass",
        vlm_has_defect   = bool(s3.get("has_defect", False)),
        vlm_confidence   = float(s3.get("confidence", 0.0)),
        caption          = s3.get("caption", "") or "",
        category         = category,
        image_path       = s1.get("image_path", ""),
    )


def _format_decision(out: DecisionOutput) -> Dict:
    PRIORITY_LABEL = {0: "OK", 1: "minor", 2: "major", 3: "critical"}
    return {
        "verdict":      out.final_verdict,
        "priority":     out.priority,
        "priority_label": PRIORITY_LABEL.get(out.priority, "unknown"),
        "confidence":   out.confidence,
        "needs_review": out.needs_review,
        "stage_votes":  out.stage_votes,
        "reasons":      out.reasons,
    }


def _pass_decision() -> Dict:
    return {
        "verdict": "Pass", "priority": 0, "priority_label": "OK",
        "confidence": 0.95, "needs_review": False,
        "stage_votes": {"stage1": "Pass"}, "reasons": ["Stage 1: Pass"],
    }


def _fail_decision_stage1_only(s1: Dict) -> Dict:
    return {
        "verdict": "Fail", "priority": 1, "priority_label": "minor",
        "confidence": 0.7, "needs_review": True,
        "stage_votes": {"stage1": "Fail"},
        "reasons": [f"Stage 1: Fail (score={s1['score']:.3f})"],
    }


if __name__ == "__main__":
    print("=== PipelineEngine smoke test ===")
    eng = PipelineEngine(category="metal_nut", stage1_only=True)
    print(f"Engine ready — stage1_only={eng.stage1_only}")
    print("(Không chạy inference thực vì chưa có checkpoint)")
    print("OK")
