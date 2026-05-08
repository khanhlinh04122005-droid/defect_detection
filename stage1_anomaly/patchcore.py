"""
stage1_anomaly/patchcore.py

PatchCore detector — kết hợp DINOv2Extractor + MemoryBank.
Đây là class chính dùng cho cả train và inference.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple
from PIL import Image

from configs.base_config import DEVICE, CHECKPOINT_DIR, CATEGORY
from configs.stage1_config import stage1_config, mvtec_tuning

from stage1_anomaly.extractor   import DINOv2Extractor, get_transform
from stage1_anomaly.memory_bank import MemoryBank


class PatchCore:
    """
    Pipeline đầy đủ:
        Ảnh → DINOv2 embeddings → MemoryBank → anomaly score map + Pass/Fail
    """

    def __init__(
        self,
        category:      str = CATEGORY,
        device:        str = DEVICE,
    ):
        self.category = category
        self.device   = device

        # Lấy tuning riêng cho category (nếu có)
        tuning = mvtec_tuning.get(category, {})

        feature_layers = tuning.get("feature_layers", stage1_config["feature_layers"])
        self.k_nearest = tuning.get("k_nearest",      stage1_config["anomaly"]["k_nearest"])
        self.thr_pct   = tuning.get("threshold_percentile", stage1_config["threshold"]["percentile"])

        # Khởi tạo components
        self.extractor = DINOv2Extractor(
            backbone      = stage1_config["backbone"],
            feature_layers= feature_layers,
            device        = device,
        )

        self.bank = MemoryBank(
            embed_dim = self.extractor.embed_dim,
            k_nearest = self.k_nearest,
        )

        self.transform = get_transform(stage1_config["train"]["image_size"])
        self.image_size = stage1_config["train"]["image_size"]

    # Train — build memory bank
    def fit(self, dataloader) -> None:
        """
        Build memory bank từ DataLoader chứa ảnh pass.

        Args:
            dataloader: yields (images, _) — images shape (B, 3, H, W)
        """
        print(f"[PatchCore] Building memory bank — category: {self.category}")
        all_embeddings = []

        self.extractor.eval()

        for batch in dataloader:
            images = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            with torch.no_grad():
                emb = self.extractor(images)   # (B, N_patches, D)

            # Flatten B × N_patches vào 1 chiều
            B, N, D = emb.shape
            all_embeddings.append(emb.reshape(B * N, D).cpu())

        all_embeddings = torch.cat(all_embeddings, dim=0)
        print(f"[PatchCore] Total patches: {all_embeddings.shape}")

        self.bank.build(all_embeddings)

    def fit_threshold(self, val_dataloader) -> float:
        """
        Tính threshold từ validation set (ảnh pass).
        Chạy sau fit().
        """
        val_scores = []

        for batch in val_dataloader:
            images = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)

            for img in images:
                _, score = self._score_tensor(img.unsqueeze(0))
                val_scores.append(score.item())

        return self.bank.fit_threshold(
            np.array(val_scores),
            method     = stage1_config["threshold"]["method"],
            percentile = self.thr_pct,
        )

    # Inference

    def predict(self, image_path: str) -> Dict:
        """
        Inference 1 ảnh từ đường dẫn file.

        Returns:
            dict với keys: image_path, image_score, prediction,
                           score_map (numpy H×W), threshold
        """
        img   = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        score_map, image_score = self._score_tensor(tensor)
        prediction = self.bank.predict(image_score.item())

        # Reshape score map về spatial grid
        n_side    = int(self.extractor.num_patches ** 0.5)
        map_2d    = score_map.reshape(n_side, n_side).numpy()
        map_resized = cv2.resize(map_2d, (self.image_size, self.image_size))

        return {
            "image_path"  : str(image_path),
            "image_score" : float(image_score),
            "prediction"  : prediction,
            "score_map"   : map_resized,    # (H, W) numpy float32
            "threshold"   : self.bank.threshold,
        }

    def _score_tensor(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score 1 ảnh dạng tensor."""
        with torch.no_grad():
            emb = self.extractor(tensor)   # (1, N_patches, D)

        score_map, image_score = self.bank.score(emb.squeeze(0))
        return score_map, image_score

    # Save / Load

    def save(self, path: str = None):
        p = path or f"{CHECKPOINT_DIR}/stage1_{self.category}"
        self.bank.save(p)
        print(f"[PatchCore] Saved → {p}")

    def load(self, path: str = None):
        p = path or f"{CHECKPOINT_DIR}/stage1_{self.category}"
        self.bank.load(p)
        print(f"[PatchCore] Loaded ← {p}")

    def __repr__(self):
        return (
            f"PatchCore(category={self.category}, "
            f"extractor={self.extractor.backbone_name}, "
            f"bank={self.bank})"
        )


if __name__ == "__main__":
    print("=== PatchCore smoke test ===")
    pc = PatchCore(category="metal_nut")
    print(pc)
    print("Init OK")