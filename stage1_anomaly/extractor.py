import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Dict
from pathlib import Path

from configs.base_config import DEVICE, IMAGE_MEAN, IMAGE_STD
from configs.stage1_config import stage1_config


class DINOv2Extractor(nn.Module):
    """
    Wrap DINOv2 ViT-L/14, hook vào các layer chỉ định,
    trả về concatenated patch embeddings.
    """

    def __init__(
        self,
        backbone: str = None,
        feature_layers: List[int] = None,
        device: str = DEVICE,
    ):
        super().__init__()

        self.backbone_name   = backbone      or stage1_config["backbone"]
        self.feature_layers  = feature_layers or stage1_config["feature_layers"]
        self.device          = device
        self._hooks          = []
        self._features: Dict[int, torch.Tensor] = {}

        self._load_backbone()
        self._register_hooks()
        self.eval()
        self.to(self.device)

    # Setup
    def _load_backbone(self):
        """Load DINOv2 từ torch.hub (tự download lần đầu)."""
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            self.backbone_name,
            pretrained=True,
        )
        # Freeze toàn bộ — chỉ dùng để extract, không train
        for p in self.model.parameters():
            p.requires_grad = False

    def _register_hooks(self):
        """Gắn forward hook vào các transformer block chỉ định."""

        def _make_hook(layer_idx: int):
            def hook(module, input, output):
                # output shape: (B, num_patches + 1, embed_dim)
                # Bỏ CLS token (index 0), giữ patch tokens
                self._features[layer_idx] = output[:, 1:, :]
            return hook

        for idx in self.feature_layers:
            h = self.model.blocks[idx].register_forward_hook(_make_hook(idx))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # Forward
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — ảnh đã normalize

        Returns:
            embeddings: (B, num_patches, embed_dim * num_layers)
        """
        self._features.clear()
        _ = self.model(x)

        # Lấy feature theo thứ tự layer, concat theo dim embedding
        layer_feats = [self._features[l] for l in self.feature_layers]
        embeddings  = torch.cat(layer_feats, dim=-1)   # (B, N_patches, D*L)

        return embeddings

    # Helpers
    @property
    def embed_dim(self) -> int:
        """Tổng chiều embedding sau concat các layer."""
        base_dim = self.model.embed_dim   # 1024 với ViT-L
        return base_dim * len(self.feature_layers)

    @property
    def num_patches(self) -> int:
        """Số patch trên 1 ảnh (với image_size=518, patch=14 → 37×37=1369)."""
        img_size   = stage1_config["train"]["image_size"]
        patch_size = stage1_config["patch_size"]
        n          = img_size // patch_size
        return n * n

    def __repr__(self):
        return (
            f"DINOv2Extractor("
            f"backbone={self.backbone_name}, "
            f"layers={self.feature_layers}, "
            f"embed_dim={self.embed_dim})"
        )


# Transform chuẩn cho DINOv2
def get_transform(image_size: int = None) -> transforms.Compose:
    """
    Transform chuẩn: resize → center crop → normalize theo ImageNet.
    image_size mặc định lấy từ stage1_config.
    """
    size = image_size or stage1_config["train"]["image_size"]

    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


# Quick test

if __name__ == "__main__":
    import torch
    from PIL import Image

    print("=== DINOv2Extractor test ===")

    extractor = DINOv2Extractor()
    print(extractor)
    print(f"Num patches : {extractor.num_patches}")
    print(f"Embed dim   : {extractor.embed_dim}")

    # Dummy input
    dummy = torch.randn(2, 3, 518, 518).to(DEVICE)
    out   = extractor(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")   # (2, 1369, 4096)
    print("OK")