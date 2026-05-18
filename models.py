"""
PigFormer and baseline models.

PigFormer
---------
Transformer encoder with Rotary Position Embedding (RoPE) and dual (mean+max)
pooling. Input is a 96x224 height map tokenized column-wise into 224 tokens of
dimension 96. Outputs three regression targets: backfat, loin, total tissue
depth at the 12th rib. See Section 3.3 of the paper.

Baselines
---------
MLP: height map flattened and passed through a three-layer MLP.
CNN: 3-channel input (height, valid-depth mask, gradient magnitude) through a
     convolutional encoder and global average pool.
RawDepthResNet: ImageNet-pretrained ResNet-18 with a 1-channel conv1 over
     raw 576x640 depth (no preprocessing baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Rotary Position Embedding
# ---------------------------------------------------------------------------

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device):
    theta = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, theta)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rope(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rope = rope_cache[: x.shape[2], :].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * rope
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class RoPEAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0, seq_len: int = 224):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "rope_cache", _build_rope_cache(seq_len, self.head_dim, torch.device("cpu"))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = _apply_rope(q, self.rope_cache)
        k = _apply_rope(k, self.rope_cache)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class RoPEEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, seq_len: int = 224):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RoPEAttention(d_model, nhead, dropout, seq_len)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def _run_encoder_layers(layers: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        x = layer(x)
    return x


def _dual_pool_tokens(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=1)


# ---------------------------------------------------------------------------
# PigFormer
# ---------------------------------------------------------------------------

class PigFormerEncoder(nn.Module):
    """Reusable PigFormer token encoder.

    The state-dict key layout is ``layers.*`` so checkpoints can be loaded into
    ``PigFormer.layers`` without renaming.
    """

    def __init__(
        self,
        seq_len: int = 224,
        feature_dim: int = 96,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(feature_dim, nhead, dim_feedforward, dropout, seq_len)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _run_encoder_layers(self.layers, x)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        return _dual_pool_tokens(tokens)

    def forward_pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.forward(x))


class PigFormer(nn.Module):
    """RoPE transformer with dual mean+max pooling. Matches the paper's
    arch_rope_pool configuration (1 layer, nhead=8, dim_ff=512, dropout=0.0).
    """

    def __init__(
        self,
        seq_len: int = 224,
        feature_dim: int = 96,
        dim_out: int = 3,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        num_layers: int = 1,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPEEncoderLayer(feature_dim, nhead, dim_feedforward, dropout, seq_len)
            for _ in range(num_layers)
        ])
        # Dual pooling: concat(mean, max) along sequence dim -> 2 * feature_dim
        pool_dim = feature_dim * 2
        self.head = nn.Sequential(
            nn.LayerNorm(pool_dim),
            nn.Linear(pool_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(dim_feedforward, dim_out),
        )

    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return _run_encoder_layers(self.layers, x)

    def pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return _dual_pool_tokens(tokens)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.encode_tokens(x)
        return tokens, self.pool_tokens(tokens)

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {"layers." + k: v for k, v in self.layers.state_dict().items()}

    def load_encoder_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True):
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            if key.startswith("encoder."):
                key = key[len("encoder."):]
            if key.startswith("online_encoder."):
                key = key[len("online_encoder."):]
            if key.startswith("layers."):
                key = key[len("layers."):]
            cleaned[key] = value
        return self.layers.load_state_dict(cleaned, strict=strict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _tokens, pooled = self.forward_features(x)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# MLP baseline
# ---------------------------------------------------------------------------

class MLPRegressor(nn.Module):
    """Per-slice MLP encoder followed by an aggregation head.

    Each cross-sectional slice (column of the height map) is passed through a
    small shared MLP to a 16-d feature, then concatenated across the sequence
    and fed to a three-layer head. Matches the MLP baseline in the paper.
    """

    def __init__(self, seq_len: int = 224, feature_dim: int = 96, dim_out: int = 3):
        super().__init__()
        self.dim_cat = 16 * seq_len
        self.encoding1 = nn.Linear(feature_dim, 128)
        self.encoding2 = nn.Linear(128, 64)
        self.encoding3 = nn.Linear(64, 16)
        self.fc1 = nn.Linear(self.dim_cat, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.encoding1(x))
        x = F.relu(self.encoding2(x))
        x = F.relu(self.encoding3(x))
        x_cat = x.reshape(-1, self.dim_cat)
        x = F.relu(self.fc1(x_cat))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


# ---------------------------------------------------------------------------
# CNN baseline
# ---------------------------------------------------------------------------

class CNNRegressor(nn.Module):
    """Vanilla CNN encoder over a 3-channel image (height, valid mask, gradient
    magnitude), followed by global average pool and an MLP head.
    """

    def __init__(self, in_channels: int = 3, dim_out: int = 3, embed_dim: int = 192, head_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
        )
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_dim),
            nn.GELU(),
            nn.Linear(head_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.head(self.encoder(x))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_model(arch: str, seq_len: int = 224, feature_dim: int = 96, dim_out: int = 3, **kwargs) -> nn.Module:
    if arch == "pigformer":
        return PigFormer(
            seq_len=seq_len,
            feature_dim=feature_dim,
            dim_out=dim_out,
            nhead=kwargs.get("nhead", 8),
            dim_feedforward=kwargs.get("dim_feedforward", 512),
            dropout=kwargs.get("dropout", 0.0),
            num_layers=kwargs.get("num_layers", 1),
            head_dropout=kwargs.get("head_dropout", 0.0),
        )
    if arch == "mlp":
        return MLPRegressor(seq_len=seq_len, feature_dim=feature_dim, dim_out=dim_out)
    if arch == "cnn":
        return CNNRegressor(in_channels=kwargs.get("in_channels", 3), dim_out=dim_out)
    raise ValueError(f"Unknown arch: {arch}. Choices: pigformer, mlp, cnn")
