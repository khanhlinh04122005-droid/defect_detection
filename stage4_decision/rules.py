"""
stage4_decision/rules.py

Bộ luật kinh doanh (rule-based) tổng hợp kết quả từ Stage 1-3
thành quyết định cuối cùng Pass/Fail + mức độ ưu tiên.

Thiết kế: đơn giản, dễ chỉnh theo từng sản phẩm/nhà máy.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


# Mức độ nghiêm trọng
SEVERITY_RANK = {
    "critical": 3,
    "major":    2,
    "minor":    1,
    "unknown":  0,
}

# Loại lỗi luôn Fail — khớp với VQA_QUESTIONS defect_type trong stage3_config.py
ALWAYS_FAIL_TYPES = {
    "hole",             # Thủng vải
    "tear",             # Rách vải
    "contamination",    # Bẩn dầu mỡ
}

# Loại lỗi Fail khi severity ≥ major
MAJOR_FAIL_TYPES = {
    "stain",            # Loang màu
    "weave",            # Lỗi dệt
    "discoloration",    # Bạc màu
    "yarn",             # Sợi thừa/thiếu (nếu major/critical)
}


@dataclass
class DecisionInput:
    """Tổng hợp output từ Stage 1, 2, 3."""
    # Stage 1
    anomaly_score:    float = 0.0
    stage1_threshold: float = 0.0
    stage1_pred:      str   = "Pass"   # "Pass" | "Fail"

    # Stage 2
    mask_iou:         float = 0.0      # 0 nếu không có mask
    mask_area_ratio:  float = 0.0      # % diện tích ảnh bị lỗi
    sam_confidence:   float = 0.0

    # Stage 3
    defect_type:      str   = "unknown"
    severity:         str   = "unknown"
    location:         str   = "unknown"
    vlm_pass_fail:    str   = "Pass"
    vlm_has_defect:   bool  = False
    vlm_confidence:   float = 0.0
    caption:          str   = ""

    # Meta
    image_path:       str   = ""
    category:         str   = ""


@dataclass
class DecisionOutput:
    """Kết quả quyết định cuối cùng."""
    final_verdict:  str     = "Pass"   # "Pass" | "Fail"
    priority:       int     = 0        # 0=OK, 1=minor, 2=major, 3=critical
    confidence:     float   = 0.0      # Độ tin cậy tổng hợp [0,1]
    reasons:        list    = field(default_factory=list)
    stage_votes:    Dict    = field(default_factory=dict)
    needs_review:   bool    = False    # True → cần người review thủ công


class DecisionEngine:
    """
    Tổng hợp kết quả 3 stage → 1 quyết định.

    Chiến lược voting:
        - Stage 1 (anomaly): 40% trọng số
        - Stage 2 (segment): 20% trọng số
        - Stage 3 (VLM):     40% trọng số
        - Nếu ALWAYS_FAIL_TYPES → Fail ngay lập tức (override)
        - Nếu 2/3 stages đồng ý → Final = majority
        - Nếu không nhất trí + confidence thấp → needs_review = True
    """

    # Ngưỡng có thể điều chỉnh
    SCORE_RATIO_THRESH  = 1.2   # anomaly_score / threshold > này → Fail
    MIN_MASK_AREA       = 0.02  # mask_area_ratio > 2% → Stage 2 vote Fail
    MIN_VLM_CONFIDENCE  = 0.4   # VLM confidence dưới ngưỡng → không đáng tin
    REVIEW_THRESHOLD    = 0.55  # confidence tổng < này → cần review

    def decide(self, inp: DecisionInput) -> DecisionOutput:
        reasons = []
        votes   = {}   # {"stage1": "Fail", "stage2": "Pass", "stage3": "Fail"}

        # Stage 1 vote
        stage1_vote, stage1_conf = self._vote_stage1(inp, reasons)
        votes["stage1"] = stage1_vote

        # Stage 2 vote
        stage2_vote, stage2_conf = self._vote_stage2(inp, reasons)
        votes["stage2"] = stage2_vote

        # Stage 3 vote
        stage3_vote, stage3_conf = self._vote_stage3(inp, reasons)
        votes["stage3"] = stage3_vote

        # Hard override: loại lỗi luôn Fail
        if inp.defect_type in ALWAYS_FAIL_TYPES and inp.vlm_has_defect:
            reasons.append(f"Override: defect_type='{inp.defect_type}' luôn là Fail")
            final    = "Fail"
            priority = 3
            conf     = 0.95
        else:
            # Weighted majority voting
            fail_weight = (
                0.6 * (stage1_vote == "Fail") +
                0.2 * (stage2_vote == "Fail") +
                0.2 * (stage3_vote == "Fail")
            )
            final    = "Fail" if fail_weight >= 0.5 else "Pass"
            conf     = max(fail_weight, 1 - fail_weight)
            priority = self._calc_priority(inp, final)

        needs_review = (
            conf < self.REVIEW_THRESHOLD or
            inp.vlm_confidence < self.MIN_VLM_CONFIDENCE
        )

        if needs_review:
            reasons.append("Độ tin cậy thấp → cần review thủ công")

        return DecisionOutput(
            final_verdict = final,
            priority      = priority,
            confidence    = round(conf, 3),
            reasons       = reasons,
            stage_votes   = votes,
            needs_review  = needs_review,
        )

    def _vote_stage1(self, inp: DecisionInput, reasons: list):
        ratio = inp.anomaly_score / max(inp.stage1_threshold, 1e-9)
        if inp.stage1_pred == "Fail" and ratio >= self.SCORE_RATIO_THRESH:
            reasons.append(f"Stage1: score={inp.anomaly_score:.3f} ({ratio:.1f}× threshold)")
            return "Fail", min(ratio / 3.0, 1.0)
        elif inp.stage1_pred == "Fail":
            reasons.append(f"Stage1: Fail (score={inp.anomaly_score:.3f}, ratio={ratio:.2f})")
            return "Fail", 0.6
        return "Pass", 0.8

    def _vote_stage2(self, inp: DecisionInput, reasons: list):
        if inp.mask_area_ratio >= self.MIN_MASK_AREA and inp.sam_confidence > 0.3:
            area_pct = inp.mask_area_ratio * 100
            reasons.append(f"Stage2: mask area={area_pct:.1f}% (>{self.MIN_MASK_AREA*100:.0f}%)")
            return "Fail", inp.sam_confidence
        return "Pass", 0.7

    def _vote_stage3(self, inp: DecisionInput, reasons: list):
        if inp.vlm_pass_fail == "Fail" and inp.vlm_confidence >= self.MIN_VLM_CONFIDENCE:
            reasons.append(
                f"Stage3: VLM=Fail, type={inp.defect_type}, "
                f"severity={inp.severity}, conf={inp.vlm_confidence:.2f}"
            )
            return "Fail", inp.vlm_confidence
        elif inp.vlm_confidence < self.MIN_VLM_CONFIDENCE:
            reasons.append(f"Stage3: VLM confidence thấp ({inp.vlm_confidence:.2f}) — bỏ qua")
            return inp.stage1_pred, 0.5   # Fallback sang Stage 1
        return "Pass", inp.vlm_confidence

    def _calc_priority(self, inp: DecisionInput, verdict: str) -> int:
        if verdict == "Pass":
            return 0
        sev = SEVERITY_RANK.get(inp.severity, 0)
        if inp.defect_type in ALWAYS_FAIL_TYPES:
            return 3
        if inp.defect_type in MAJOR_FAIL_TYPES:
            return max(sev, 2)
        return max(sev, 1)


# Singleton — dùng chung toàn dự án
_engine = DecisionEngine()


def decide(inp: DecisionInput) -> DecisionOutput:
    """Shortcut để gọi DecisionEngine từ bên ngoài."""
    return _engine.decide(inp)


if __name__ == "__main__":
    print("=== Rule engine smoke test ===")

    sample = DecisionInput(
        anomaly_score    = 0.85,
        stage1_threshold = 0.40,
        stage1_pred      = "Fail",
        mask_area_ratio  = 0.05,
        sam_confidence   = 0.75,
        defect_type      = "scratch",
        severity         = "major",
        vlm_pass_fail    = "Fail",
        vlm_has_defect   = True,
        vlm_confidence   = 0.82,
        category         = "metal_nut",
    )

    out = decide(sample)
    print(f"Verdict   : {out.final_verdict}")
    print(f"Priority  : {out.priority}")
    print(f"Confidence: {out.confidence}")
    print(f"Votes     : {out.stage_votes}")
    print(f"Review?   : {out.needs_review}")
    print(f"Reasons   :")
    for r in out.reasons:
        print(f"  - {r}")
