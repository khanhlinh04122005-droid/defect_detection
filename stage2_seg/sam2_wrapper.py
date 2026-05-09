import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from PIL import Image

from configs.base_config import DEVICE, CHECKPOINT_DIR, CATEGORY
from configs.stage2_config import (
    SAM2_WEIGHTS, SAM2_CONFIG,
    LORA_WEIGHTS_PATH,
    PROMPT_TYPE, TOPK_POINTS, POINT_LABEL, PROMPT_THRESHOLD,
)
from stage2_seg.lora_adapter import inject_lora, save_lora_weights, load_lora_weights


class SAM2Wrapper(nn.Module):
    """
    Wrapper quanh SAM2 image encoder + mask decoder.

    Tính năng:
        - Load SAM2 từ checkpoint chính thức
        - Inject LoRA vào image encoder (attention projections)
        - Nhận anomaly score map từ Stage 1 → tạo point / box prompt
        - Trả về binary mask vùng lỗi
    """

    def __init__(
        self,
        device: str     = DEVICE,
        use_lora: bool  = True,
        lora_weights: str = None,
    ):
        super().__init__()
        self.device      = device
        self.use_lora    = use_lora

        self._load_sam2()

        if use_lora:
            inject_lora(self.image_encoder)

        # Load LoRA weights nếu có sẵn
        weights_path = lora_weights or LORA_WEIGHTS_PATH
        if Path(weights_path).exists():
            load_lora_weights(self.image_encoder, weights_path)
            print(f"[SAM2Wrapper] LoRA weights loaded ← {weights_path}")
        else:
            print("[SAM2Wrapper] No LoRA weights found — using zero-shot SAM2")

        self.to(device)

    def _load_sam2(self):
        """Load SAM2 với fp16 + gradient checkpointing để fit 4GB VRAM."""
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            torch.cuda.empty_cache()

            sam2_model = build_sam2(SAM2_CONFIG, SAM2_WEIGHTS, device=self.device)

            # Không .half() thủ công — dùng autocast trong encode_for_training

            # Gradient checkpointing: đổi activation memory lấy compute time
            if hasattr(sam2_model.image_encoder.trunk, "use_checkpoint"):
                sam2_model.image_encoder.trunk.use_checkpoint = True

            self.predictor     = SAM2ImagePredictor(sam2_model)
            self.image_encoder = sam2_model.image_encoder
            print(f"[SAM2Wrapper] SAM2 loaded (bfloat16 autocast) — config: {SAM2_CONFIG}")

        except ImportError:
            raise ImportError(
                "Chưa cài sam2. Chạy:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2"
            )
        except Exception as e:
            raise RuntimeError(f"[SAM2Wrapper] Lỗi load SAM2: {e}")

    def score_map_to_prompt(
        self,
        score_map: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> Dict:
        """
        Chuyển anomaly score map (H×W float32) từ Stage 1
        thành prompt cho SAM2 (points hoặc box).

        Args:
            score_map: (H, W) — anomaly score từ PatchCore.predict()
            image_hw:  (H, W) — kích thước ảnh gốc

        Returns:
            dict với keys "point_coords", "point_labels" hoặc "box"
        """
        h, w = image_hw
        map_h, map_w = score_map.shape

        # Normalize score_map về [0, 1]
        s_min, s_max = score_map.min(), score_map.max()
        if s_max > s_min:
            norm = (score_map - s_min) / (s_max - s_min)
        else:
            norm = score_map

        prompt = {}

        if PROMPT_TYPE in ("point", "both"):
            # Lấy top-K pixel có score cao nhất
            flat_idx  = np.argsort(norm.ravel())[::-1]
            top_idx   = []
            for i in flat_idx:
                if norm.ravel()[i] < PROMPT_THRESHOLD:
                    break
                top_idx.append(i)
                if len(top_idx) >= TOPK_POINTS:
                    break

            if top_idx:
                ys, xs = np.unravel_index(top_idx, (map_h, map_w))
                # Scale tọa độ về kích thước ảnh gốc
                xs_orig = (xs / map_w * w).astype(int)
                ys_orig = (ys / map_h * h).astype(int)
                prompt["point_coords"] = np.stack([xs_orig, ys_orig], axis=1)  # (K, 2)
                prompt["point_labels"] = np.full(len(top_idx), POINT_LABEL, dtype=int)

        if PROMPT_TYPE in ("box", "both"):
            # Bounding box của vùng score > threshold
            mask_thresh = norm > PROMPT_THRESHOLD
            if mask_thresh.any():
                rows = np.any(mask_thresh, axis=1)
                cols = np.any(mask_thresh, axis=0)
                r_min, r_max = np.where(rows)[0][[0, -1]]
                c_min, c_max = np.where(cols)[0][[0, -1]]
                # Scale về ảnh gốc
                x0 = int(c_min / map_w * w)
                y0 = int(r_min / map_h * h)
                x1 = int(c_max / map_w * w)
                y1 = int(r_max / map_h * h)
                prompt["box"] = np.array([x0, y0, x1, y1])

        return prompt

    def encode_for_training(self, image_np: np.ndarray):
        """
        Encode image với gradient (dùng trong training loop).
        Bypass SAM2ImagePredictor.set_image() vì nó chạy @no_grad.

        Returns:
            features: dict với 'image_embed' (1,C,H,W) và 'high_res_feats' (list)
            img_pe:   dense positional encoding từ prompt encoder
        """
        sam_model = self.predictor.model

        img_t = self.predictor._transforms(image_np).unsqueeze(0).to(self.device)

        # autocast: PyTorch tự xử lý fp16 cho image encoder (tiết kiệm VRAM)
        # Không dùng .half() thủ công vì forward_image gọi cả conv trong mask decoder
        # forward_image trong autocast context của caller (train.py)
        backbone_out = sam_model.forward_image(img_t)
        _, vision_feats, _, _ = sam_model._prepare_backbone_features(backbone_out)

        bb_feat_sizes = [(256, 256), (128, 128), (64, 64)]
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
        ][::-1]

        features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        img_pe   = sam_model.sam_prompt_encoder.get_dense_pe()
        return features, img_pe

    @torch.no_grad()
    def predict_mask(
        self,
        image: np.ndarray,
        score_map: np.ndarray,
        multimask: bool = False,
    ) -> Tuple[np.ndarray, float]:
        """
        Segment vùng lỗi từ 1 ảnh + anomaly score map.

        Args:
            image:     (H, W, 3) uint8 RGB
            score_map: (H, W) float32 — từ Stage 1
            multimask: nếu True, SAM2 trả về 3 mask, lấy mask tốt nhất

        Returns:
            mask:  (H, W) bool
            score: float — độ tin cậy của SAM2
        """
        h, w = image.shape[:2]
        prompt = self.score_map_to_prompt(score_map, (h, w))

        if not prompt:
            # Không có vùng bất thường đủ mạnh → trả về mask rỗng
            return np.zeros((h, w), dtype=bool), 0.0

        self.predictor.set_image(image)

        masks, scores, _ = self.predictor.predict(
            point_coords  = prompt.get("point_coords"),
            point_labels  = prompt.get("point_labels"),
            box           = prompt.get("box"),
            multimask_output = multimask,
        )

        # Chọn mask có score cao nhất
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), float(scores[best_idx])

    def forward(
        self,
        image: np.ndarray,
        score_map: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Alias của predict_mask để dùng trong training loop."""
        return self.predict_mask(image, score_map)

    def save_lora(self, path: str = None):
        p = path or LORA_WEIGHTS_PATH
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        save_lora_weights(self.image_encoder, p)

    def load_lora(self, path: str = None):
        p = path or LORA_WEIGHTS_PATH
        load_lora_weights(self.image_encoder, p)


if __name__ == "__main__":
    print("=== SAM2Wrapper smoke test (prompt only, không load SAM2) ===")

    # Test score_map_to_prompt không cần SAM2
    wrapper = object.__new__(SAM2Wrapper)
    wrapper.device = "cpu"

    dummy_score = np.random.rand(28, 28).astype(np.float32)
    dummy_score[10:18, 10:18] = 0.9  # Vùng lỗi giả

    prompt = wrapper.score_map_to_prompt(dummy_score, (224, 224))
    print(f"Prompt type  : {PROMPT_TYPE}")
    if "point_coords" in prompt:
        print(f"Point coords : {prompt['point_coords']}")
        print(f"Point labels : {prompt['point_labels']}")
    if "box" in prompt:
        print(f"Box          : {prompt['box']}")
    print("OK")
