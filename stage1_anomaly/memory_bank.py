"""
stage1_anomaly/memory_bank.py

FAISS memory bank với coreset subsampling.
Lưu patch embeddings của ảnh pass, dùng để tính anomaly score tại inference.
"""

import torch
import numpy as np
import faiss
import json
from pathlib import Path
from typing import Optional, Tuple

from configs.base_config import DEVICE
from configs.stage1_config import stage1_config


class MemoryBank:
    """
    Lưu coreset của patch embeddings từ ảnh pass.
    Tại inference: tính khoảng cách nearest-neighbour → anomaly score.
    """

    def __init__(
        self,
        embed_dim: int,
        coreset_ratio: float = None,
        max_samples:   int   = None,
        k_nearest:     int   = None,
        use_gpu:       bool  = True,
    ):
        cfg_mem = stage1_config["memory_bank"]
        cfg_ano = stage1_config["anomaly"]

        self.embed_dim     = embed_dim
        self.coreset_ratio = coreset_ratio or cfg_mem["coreset_ratio"]
        self.max_samples   = max_samples   or cfg_mem["max_samples"]
        self.k_nearest     = k_nearest     or cfg_ano["k_nearest"]
        self.aggregation   = cfg_ano["score_aggregation"]   # "max" | "mean"
        self.use_gpu       = use_gpu and torch.cuda.is_available()

        self.bank: Optional[np.ndarray] = None   # (N, D) float32
        self.index = None                         # faiss index
        self.threshold: Optional[float] = None

    # Build

    def build(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings: (N_images * N_patches, D) — tất cả patch embeddings
                        từ ảnh pass trong train set.
        """
        vectors = embeddings.cpu().numpy().astype(np.float32)

        print(f"[MemoryBank] Raw vectors: {vectors.shape}")

        # Coreset subsampling — giữ lại subset đại diện
        sampled = self._coreset_sample(vectors)
        print(f"[MemoryBank] After coreset: {sampled.shape}")

        self.bank = sampled
        self._build_index()

    def _coreset_sample(self, vectors: np.ndarray) -> np.ndarray:
        """
        Greedy coreset: chọn subset sao cho coverage tốt nhất.
        Nếu số lượng nhỏ hơn ngưỡng thì giữ nguyên.
        """
        n_total  = len(vectors)
        n_target = min(
            int(n_total * self.coreset_ratio),
            self.max_samples,
        )

        if n_target >= n_total:
            return vectors

        # Random init — chọn 1 điểm đầu tiên ngẫu nhiên
        rng      = np.random.default_rng(42)
        selected = [rng.integers(n_total)]
        dists    = np.full(n_total, np.inf)

        for _ in range(n_target - 1):
            # Khoảng cách từ mỗi điểm đến điểm đã chọn gần nhất
            last    = vectors[selected[-1]]
            d       = np.linalg.norm(vectors - last, axis=1)
            dists   = np.minimum(dists, d)
            selected.append(int(np.argmax(dists)))

        return vectors[selected]

    def _build_index(self):
        """Tạo FAISS index (GPU nếu có, CPU fallback)."""
        d     = self.embed_dim
        index = faiss.IndexFlatL2(d)

        if self.use_gpu:
            res         = faiss.StandardGpuResources()
            self.index  = faiss.index_cpu_to_gpu(res, 0, index)
        else:
            self.index  = index

        self.index.add(self.bank)
        print(f"[MemoryBank] FAISS index built — {self.index.ntotal} vectors")


    # Inference

    def score(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tính anomaly score cho 1 ảnh.

        Args:
            embeddings: (N_patches, D) — patch embeddings của 1 ảnh

        Returns:
            score_map : (N_patches,)  — anomaly score per patch
            image_score: scalar       — score của toàn ảnh
        """
        vectors = embeddings.cpu().numpy().astype(np.float32)

        # K nearest-neighbour distances
        distances, _ = self.index.search(vectors, self.k_nearest)  # (N, K)

        # Aggregate K distances thành 1 score per patch
        patch_scores = distances[:, 0]   # dùng nearest-1 cho map
        knn_scores   = distances.mean(axis=1)  # mean K cho stability

        score_map = torch.from_numpy(knn_scores)

        # Image-level score
        if self.aggregation == "max":
            image_score = score_map.max()
        else:
            image_score = score_map.mean()

        return score_map, image_score


    # Threshold

    def fit_threshold(self, val_scores: np.ndarray, method: str = None, percentile: float = None):
        """
        Tính threshold từ validation scores (ảnh pass).

        Args:
            val_scores: array anomaly scores của ảnh pass trong val set
        """
        cfg_thr    = stage1_config["threshold"]
        method     = method     or cfg_thr["method"]
        percentile = percentile or cfg_thr["percentile"]

        if method == "percentile":
            self.threshold = float(np.percentile(val_scores, percentile))
        else:
            # Mean + 3 std
            self.threshold = float(val_scores.mean() + 3 * val_scores.std())

        print(f"[MemoryBank] Threshold = {self.threshold:.4f} (method={method})")
        return self.threshold

    def predict(self, image_score: float) -> str:
        """Pass / Fail dựa trên threshold."""
        assert self.threshold is not None, "Chưa fit threshold!"
        return "Fail" if image_score > self.threshold else "Pass"

  
    # Save / Load

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Lưu bank numpy + threshold
        np.save(str(path) + "_bank.npy", self.bank)

        meta = {
            "embed_dim"    : self.embed_dim,
            "coreset_ratio": self.coreset_ratio,
            "k_nearest"    : self.k_nearest,
            "aggregation"  : self.aggregation,
            "threshold"    : self.threshold,
            "n_vectors"    : len(self.bank),
        }
        with open(str(path) + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[MemoryBank] Saved → {path}")

    def load(self, path: str):
        path = Path(path)

        self.bank = np.load(str(path) + "_bank.npy")

        with open(str(path) + "_meta.json") as f:
            meta = json.load(f)

        self.embed_dim     = meta["embed_dim"]
        self.coreset_ratio = meta["coreset_ratio"]
        self.k_nearest     = meta["k_nearest"]
        self.aggregation   = meta["aggregation"]
        self.threshold     = meta["threshold"]

        self._build_index()
        print(f"[MemoryBank] Loaded ← {path} ({self.index.ntotal} vectors)")

    def __repr__(self):
        n = self.index.ntotal if self.index else 0
        return (
            f"MemoryBank(embed_dim={self.embed_dim}, "
            f"vectors={n}, k={self.k_nearest}, "
            f"threshold={self.threshold})"
        )


if __name__ == "__main__":
    print("=== MemoryBank test ===")
    bank = MemoryBank(embed_dim=4096)

    # Dummy: 100 ảnh × 1369 patches × 4096 dim (nhỏ để test nhanh)
    dummy = torch.randn(100 * 10, 64)
    bank.embed_dim = 64
    bank.build(dummy)

    # Score 1 ảnh
    test_emb    = torch.randn(10, 64)
    smap, score = bank.score(test_emb)
    print(f"Score map : {smap.shape}")
    print(f"Image score: {score.item():.4f}")
    print("OK")