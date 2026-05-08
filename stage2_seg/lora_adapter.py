import math
import torch
import torch.nn as nn
from typing import List

from configs.stage2_config import LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES


class LoRALinear(nn.Module):
    """
    Thay thế nn.Linear bằng W + (alpha/r) * B @ A.
    A, B là ma trận thấp chiều được train — W gốc bị freeze.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()

        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.rank         = rank
        self.scale        = alpha / rank

        # Giữ nguyên weight gốc, freeze
        self.weight = linear.weight
        self.bias   = linear.bias
        for p in [self.weight] + ([self.bias] if self.bias is not None else []):
            p.requires_grad = False

        # LoRA matrices
        self.lora_A   = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B   = nn.Parameter(torch.zeros(self.out_features, rank))
        self.dropout  = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base   = nn.functional.linear(x, self.weight, self.bias)
        lora   = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + self.scale * lora


def inject_lora(
    model: nn.Module,
    target_modules: List[str] = None,
    rank: int    = LORA_RANK,
    alpha: float = LORA_ALPHA,
    dropout: float = LORA_DROPOUT,
) -> nn.Module:
    """
    Duyệt qua model, thay mọi nn.Linear có tên khớp target_modules
    bằng LoRALinear tương ứng.

    Args:
        model:          SAM2 image encoder (hoặc toàn bộ SAM2)
        target_modules: danh sách tên module cần inject, e.g. ["q_proj","v_proj"]
        rank, alpha, dropout: siêu tham số LoRA

    Returns:
        model đã được inject (in-place)
    """
    targets = target_modules or LORA_TARGET_MODULES

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Kiểm tra tên leaf module
        leaf_name = name.split(".")[-1]
        if leaf_name not in targets:
            continue

        # Tìm parent module để setattr
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], lora_layer)
        replaced += 1

    print(f"[LoRA] Injected {replaced} LoRALinear layers (rank={rank}, alpha={alpha})")
    return model


def get_lora_params(model: nn.Module):
    """Trả về chỉ các param LoRA (trainable)."""
    return [p for p in model.parameters() if p.requires_grad]


def save_lora_weights(model: nn.Module, path: str):
    """Chỉ lưu weight LoRA (A, B) — nhỏ hơn nhiều so với full model."""
    lora_state = {
        k: v for k, v in model.state_dict().items()
        if "lora_A" in k or "lora_B" in k
    }
    torch.save(lora_state, path)
    n_params = sum(v.numel() for v in lora_state.values())
    print(f"[LoRA] Saved {len(lora_state)} tensors ({n_params:,} params) → {path}")


def load_lora_weights(model: nn.Module, path: str, strict: bool = False):
    """Load chỉ LoRA weights vào model đã inject."""
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"[LoRA] Loaded ← {path} | missing={len(missing)}, unexpected={len(unexpected)}")
    return model


if __name__ == "__main__":
    # Smoke test với linear nhỏ
    linear = nn.Linear(64, 32)
    lora   = LoRALinear(linear, rank=4, alpha=8.0, dropout=0.0)
    x      = torch.randn(2, 64)
    out    = lora(x)
    print(f"LoRALinear output: {out.shape}")   # (2, 32)
    print("OK")
