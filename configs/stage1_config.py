class Stage1Config:
    def __init__(self):
        self.backbone = "dinov2_vitl14"
        self.patch_size = 14
        self.feature_layers = [9, 12, 15, 18]
        self.patch_stride = 1

        self.memory_bank = {
            "coreset_ratio": 0.02,   # 2% → ~10K samples
            "max_samples": 10000     # Hard cap 10K samples
        }

        self.anomaly = {
            "k_nearest": 9,
            "score_aggregation": "max"
        }

        self.threshold = {
            "method": "percentile",
            "percentile": 99.0
        }

        self.train = {
            "image_size": 224,   # Giảm từ 518 → 224 để chạy trên CPU
            "batch_size": 4,     # Giảm từ 8 → 4 để tiết kiệm RAM
            "num_workers": 0     # Windows CPU: luôn dùng 0
        }

    def apply_tuning(self, category: str):
        """Áp dụng per-category tuning cho cả TSfabrics và MVTec."""
        if category in category_tuning:
            tuning = category_tuning[category]
            self.feature_layers = tuning["feature_layers"]
            self.anomaly["k_nearest"] = tuning["k_nearest"]
            self.threshold["percentile"] = tuning["threshold_percentile"]


# Per-category tuning — Tất cả các loại vải hỗ trợ
category_tuning = {
    # === TSfabrics (dataset chính) ===
    # Vải T1 — dệt thoi, 93K frames. k_nearest=9, percentile=99.0 (thực nghiệm tốt cho video-frame)
    "tsfabric_T1": {"feature_layers": [12, 18], "k_nearest": 9,  "threshold_percentile": 88.0},
    # Vải T2 — cấu trúc dệt khác, k=11 để bắt nhiều neighbour hơn
    "tsfabric_T2": {"feature_layers": [9, 12, 15, 18], "k_nearest": 11, "threshold_percentile": 99.0},
    # Vải T3 — dùng layer thấp hơn [6,9,12,15] để nắm texture mịn hơn
    "tsfabric_T3": {"feature_layers": [6, 9, 12, 15],  "k_nearest": 9,  "threshold_percentile": 98.5},

    # === Vải MVTec (giữ lại để tương thích nếu cần) ===
    "carpet":     {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.5},
    "leather":    {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.5},
    "tile":       {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.5},
    "grid":       {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.5},
    "wood":       {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.5},
    "zipper":     {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.0},
    "cotton":     {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.0},
    "denim":      {"feature_layers": [9, 12, 15, 18], "k_nearest": 11, "threshold_percentile": 99.0},
    "silk":       {"feature_layers": [6, 9, 12, 15],  "k_nearest": 7,  "threshold_percentile": 98.5},
    "synthetic":  {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.0},
    "wool":       {"feature_layers": [9, 12, 15, 18], "k_nearest": 11, "threshold_percentile": 99.5},
    "knit":       {"feature_layers": [9, 12, 15, 18], "k_nearest": 9,  "threshold_percentile": 99.0},
}

# Alias để tương thích code cũ
mvtec_tuning = category_tuning

# Dict-style config (tương thích với code import stage1_config["..."])
_cfg = Stage1Config()
stage1_config = {
    "backbone":       _cfg.backbone,
    "patch_size":     _cfg.patch_size,
    "feature_layers": _cfg.feature_layers,
    "patch_stride":   _cfg.patch_stride,
    "memory_bank":    _cfg.memory_bank,
    "anomaly":        _cfg.anomaly,
    "threshold":      _cfg.threshold,
    "train":          _cfg.train,
}