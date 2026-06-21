import torch
from pathlib import Path
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_2bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 3).round().to(torch.int8)
    x_quant_2 = (
        (x_quant_8[:, 0::4] & 0x3) |
        ((x_quant_8[:, 1::4] & 0x3) << 2) |
        ((x_quant_8[:, 2::4] & 0x3) << 4) |
        ((x_quant_8[:, 3::4] & 0x3) << 6)
    )
    return x_quant_2, normalization.to(torch.float16)


def block_dequantize_2bit(x_quant_2: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    assert x_quant_2.dim() == 2

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_2.new_empty(x_quant_2.size(0), x_quant_2.shape[1] * 4)
    x_quant_8[:, 0::4] = x_quant_2 & 0x3
    x_quant_8[:, 1::4] = (x_quant_2 >> 2) & 0x3
    x_quant_8[:, 2::4] = (x_quant_2 >> 4) & 0x3
    x_quant_8[:, 3::4] = (x_quant_2 >> 6) & 0x3
    x_norm = x_quant_8.to(torch.float32) / 3
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    assert group_size % 8 == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    v = (x_norm * 7).round().to(torch.int32).view(x.size(0), -1, 8)
    byte0 = (v[..., 0] & 7) | ((v[..., 1] & 7) << 3) | ((v[..., 2] & 3) << 6)
    byte1 = ((v[..., 2] >> 2) & 1) | ((v[..., 3] & 7) << 1) | ((v[..., 4] & 7) << 4) | ((v[..., 5] & 1) << 7)
    byte2 = ((v[..., 5] >> 1) & 3) | ((v[..., 6] & 7) << 2) | ((v[..., 7] & 7) << 5)
    x_quant_3 = torch.stack([byte0, byte1, byte2], dim=-1).reshape(x.size(0), -1).to(torch.int8)
    return x_quant_3, normalization.to(torch.float16)


def block_dequantize_3bit(x_quant_3: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    assert x_quant_3.dim() == 2

    normalization = normalization.to(torch.float32)
    b = x_quant_3.view(x_quant_3.size(0), -1, 3).to(torch.int32)
    b0, b1, b2 = b[..., 0], b[..., 1], b[..., 2]
    v0 = b0 & 7
    v1 = (b0 >> 3) & 7
    v2 = ((b0 >> 6) & 3) | ((b1 & 1) << 2)
    v3 = (b1 >> 1) & 7
    v4 = (b1 >> 4) & 7
    v5 = ((b1 >> 7) & 1) | ((b2 & 3) << 1)
    v6 = (b2 >> 2) & 7
    v7 = (b2 >> 5) & 7
    x_quant_8 = torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], dim=-1).reshape(x_quant_3.size(0), -1)
    x_norm = x_quant_8.to(torch.float32) / 7
    return ((x_norm * 2 * normalization) - normalization).view(-1)


class Linear2Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        self.register_buffer(
            "weight_q2",
            torch.zeros(out_features * in_features // group_size, group_size // 4, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(Linear2Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            weight_q2, weight_norm = block_quantize_2bit(weight.view(-1), self._group_size)
            self.weight_q2.copy_(weight_q2)
            self.weight_norm.copy_(weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_2bit(self.weight_q2, self.weight_norm).view(self._shape)
            return torch.nn.functional.linear(x, weight, self.bias)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // group_size, group_size * 3 // 8, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            weight_q3, weight_norm = block_quantize_3bit(weight.view(-1), self._group_size)
            self.weight_q3.copy_(weight_q3)
            self.weight_norm.copy_(weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_3bit(self.weight_q3, self.weight_norm).view(self._shape)
            return torch.nn.functional.linear(x, weight, self.bias)


class BigNet2Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
                torch.nn.ReLU(),
                Linear2Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
class BigNet3Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet3Bit:
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
