from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import sqrt
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as nn_functional

from luxai2021.imitation.actions import ACTION_SCHEMA, FIRST_PLACE_ACTION_SCHEMA, TARGET_NAMES
from luxai2021.imitation.data import IGNORE_INDEX
from luxai2021.imitation.masking import LEGAL_MASK_SUFFIX, apply_legal_action_mask
from luxai2021.imitation.schema import (
    BOARD_SIZES,
    CYCLE_LENGTH,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    GAME_PHASE_COUNT,
    SPATIAL_FEATURE_NAMES,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

CHECKPOINT_SCHEMA_VERSION = 1
POLICY_SCHEMA_FACTORIZED = "factorized_v1"
POLICY_SCHEMA_FIRST_PLACE_FLAT = "first_place_flat_v1"
ENCODER_TYPES = (
    "unet",
    "resnet17x32",
    "resnet17x48",
    "resattn8",
    "transformer16",
    "axial32",
    "axial32_4m5",
)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.activation = nn.SiLU()

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        residual = self.activation(self.norm1(self.conv1(inputs))) * mask
        residual = self.norm2(self.conv2(residual)) * mask
        return self.activation(inputs + residual) * mask


class _EncoderStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(output_channels), output_channels)
        self.activation = nn.SiLU()
        self.blocks = nn.ModuleList((ResidualBlock(output_channels), ResidualBlock(output_channels)))

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        output = self.activation(self.norm(self.conv(inputs))) * mask
        for block in self.blocks:
            output = block(output, mask)
        return output


@dataclass(frozen=True)
class ModelConfig:
    input_channels: int = len(FEATURE_NAMES)
    base_channels: int = 64
    feature_channels: int = 128
    cycle_embedding_dim: int = 8
    phase_embedding_dim: int = 4
    board_size_embedding_dim: int = 4
    encoder_type: str = "unet"
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_ffn_dim: int = 1024
    transformer_dropout: float = 0.1
    transformer16_layers: int = 8
    axial32_layers: int = 6
    axial32_4m5_dim: int = 192
    axial32_4m5_ffn_dim: int = 672
    axial32_4m5_layers: int = 6
    resattn8_base_channels: int = 48
    resattn8_feature_channels: int = 96
    resattn8_heads: int = 6
    resattn8_ffn_dim: int = 384
    resattn8_layers: int = 2
    policy_schema: str = POLICY_SCHEMA_FACTORIZED

    def __post_init__(self) -> None:  # noqa: C901
        if self.encoder_type not in ENCODER_TYPES:
            msg = f"Unsupported encoder type: {self.encoder_type}"
            raise ValueError(msg)
        if self.policy_schema not in {POLICY_SCHEMA_FACTORIZED, POLICY_SCHEMA_FIRST_PLACE_FLAT}:
            msg = f"Unsupported policy schema: {self.policy_schema}"
            raise ValueError(msg)
        if self.transformer_heads < 1:
            raise ValueError("transformer_heads must be positive")
        if self.transformer_dim % self.transformer_heads:
            raise ValueError("transformer_dim must be divisible by transformer_heads")
        if self.axial32_4m5_dim % self.transformer_heads:
            raise ValueError("axial32_4m5_dim must be divisible by transformer_heads")
        if self.resattn8_base_channels < 1 or self.resattn8_feature_channels < 1:
            raise ValueError("ResAttnUNet8 channel counts must be positive")
        if self.resattn8_heads < 1:
            raise ValueError("resattn8_heads must be positive")
        if (self.resattn8_base_channels * 4) % self.resattn8_heads:
            raise ValueError("The ResAttnUNet8 bottleneck width must be divisible by resattn8_heads")
        if self.transformer_ffn_dim < 1 or self.axial32_4m5_ffn_dim < 1:
            raise ValueError("Transformer FFN dimensions must be positive")
        if self.resattn8_ffn_dim < 1:
            raise ValueError("resattn8_ffn_dim must be positive")
        if not 0 <= self.transformer_dropout < 1:
            raise ValueError("transformer_dropout must be in [0, 1)")
        if (
            self.transformer16_layers < 1
            or self.axial32_layers < 1
            or self.axial32_4m5_layers < 1
            or self.resattn8_layers < 1
        ):
            raise ValueError("Transformer layer counts must be positive")


@dataclass(frozen=True)
class EncoderFeatures:
    spatial: Tensor
    global_features: Tensor


class LuxSpatialEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.input_channels != len(FEATURE_NAMES):
            msg = f"Expected {len(FEATURE_NAMES)} input channels, got {config.input_channels}"
            raise ValueError(msg)
        base = config.base_channels
        self.cycle_embedding = nn.Embedding(CYCLE_LENGTH, config.cycle_embedding_dim)
        self.phase_embedding = nn.Embedding(GAME_PHASE_COUNT, config.phase_embedding_dim)
        self.board_size_embedding = nn.Embedding(len(BOARD_SIZES), config.board_size_embedding_dim)
        categorical_channels = config.cycle_embedding_dim + config.phase_embedding_dim + config.board_size_embedding_dim
        stem_channels = len(SPATIAL_FEATURE_NAMES) + categorical_channels
        self.stem = _EncoderStage(stem_channels, base)
        self.down1 = _EncoderStage(base, base * 2, stride=2)
        self.down2 = _EncoderStage(base * 2, base * 4, stride=2)
        self.bottleneck = nn.ModuleList((ResidualBlock(base * 4), ResidualBlock(base * 4)))
        self.up1 = _EncoderStage(base * 4 + base * 2, base * 2)
        self.up2 = _EncoderStage(base * 2 + base, config.feature_channels)
        self.output_channels = config.feature_channels
        self.global_output_channels = config.feature_channels * 2 + categorical_channels
        self.register_buffer(
            "spatial_indices",
            torch.tensor([FEATURE_INDEX[name] for name in SPATIAL_FEATURE_NAMES]),
            persistent=False,
        )

    @staticmethod
    def _category_index(inputs: Tensor, name: str, category_count: int) -> Tensor:
        values = inputs[:, FEATURE_INDEX[name]].amax(dim=(-2, -1))
        return values.round().long().clamp_(0, category_count - 1)

    def forward_features(self, inputs: Tensor) -> EncoderFeatures:
        mask0 = inputs[:, FEATURE_INDEX["board_mask"] : FEATURE_INDEX["board_mask"] + 1]
        spatial_inputs = inputs.index_select(1, self.spatial_indices)
        categorical = torch.cat(
            (
                self.cycle_embedding(self._category_index(inputs, "day_night_cycle", CYCLE_LENGTH)),
                self.phase_embedding(self._category_index(inputs, "game_phase", GAME_PHASE_COUNT)),
                self.board_size_embedding(self._category_index(inputs, "board_size", len(BOARD_SIZES))),
            ),
            dim=1,
        )
        categorical_map = categorical[:, :, None, None].expand(-1, -1, inputs.shape[-2], inputs.shape[-1])
        stem_inputs = torch.cat((spatial_inputs, categorical_map), dim=1) * mask0
        level0 = self.stem(stem_inputs, mask0)

        mask1 = nn_functional.interpolate(mask0, scale_factor=0.5, mode="nearest")
        level1 = self.down1(level0, mask1)
        mask2 = nn_functional.interpolate(mask1, scale_factor=0.5, mode="nearest")
        level2 = self.down2(level1, mask2)
        for block in self.bottleneck:
            level2 = block(level2, mask2)
        up1 = nn_functional.interpolate(level2, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        up1 = self.up1(torch.cat((up1, level1), dim=1), mask1)
        up2 = nn_functional.interpolate(up1, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        spatial = self.up2(torch.cat((up2, level0), dim=1), mask0)

        valid_count = mask0.sum(dim=(-2, -1)).clamp_min(1)
        pooled_average = (spatial * mask0).sum(dim=(-2, -1)) / valid_count
        pooled_maximum = spatial.masked_fill(mask0 <= 0, torch.finfo(spatial.dtype).min).amax(dim=(-2, -1))
        global_features = torch.cat((pooled_average, pooled_maximum, categorical), dim=1)
        return EncoderFeatures(spatial=spatial, global_features=global_features)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_features(inputs).spatial


class GlobalSpatialAttentionBlock(nn.Module):
    """Global SDPA at the 8x8 U-Net bottleneck with masked residual updates."""

    def __init__(self, channels: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.position = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.attention_norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        self.projection_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )
        self.attention_dropout = dropout

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        batch_size, channels, height, width = inputs.shape
        masked_inputs = inputs * mask
        output = (masked_inputs + self.position(masked_inputs)) * mask
        tokens = output.flatten(2).transpose(1, 2)
        normalized = self.attention_norm(tokens)
        qkv = self.qkv(normalized).reshape(
            batch_size,
            height * width,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        valid_tokens = mask.flatten(1).bool()
        empty_samples = ~valid_tokens.any(dim=1)
        first_token = torch.arange(height * width, device=inputs.device) == 0
        safe_tokens = valid_tokens | (empty_samples[:, None] & first_token[None])
        attended = nn_functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=safe_tokens[:, None, None, :],
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, height * width, channels)
        tokens = tokens + self.projection_dropout(self.projection(attended))
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens.transpose(1, 2).reshape(batch_size, channels, height, width) * mask


class ResAttnUNet8SpatialEncoder(LuxSpatialEncoder):
    """Compact residual U-Net with global attention restricted to 8x8 features."""

    def __init__(self, config: ModelConfig) -> None:
        compact_config = replace(
            config,
            base_channels=config.resattn8_base_channels,
            feature_channels=config.resattn8_feature_channels,
        )
        super().__init__(compact_config)
        bottleneck_channels = config.resattn8_base_channels * 4
        blocks: list[nn.Module] = [ResidualBlock(bottleneck_channels)]
        blocks.extend(
            GlobalSpatialAttentionBlock(
                bottleneck_channels,
                config.resattn8_heads,
                config.resattn8_ffn_dim,
                config.transformer_dropout,
            )
            for _ in range(config.resattn8_layers)
        )
        self.bottleneck = nn.ModuleList(blocks)


class ResNet17SpatialEncoder(nn.Module):
    """RLIAYN-inspired full-resolution ResNet with 17 residual blocks."""

    BLOCK_COUNT = 17

    def __init__(self, config: ModelConfig, channels: int) -> None:
        super().__init__()
        if config.input_channels != len(FEATURE_NAMES):
            msg = f"Expected {len(FEATURE_NAMES)} input channels, got {config.input_channels}"
            raise ValueError(msg)
        self.cycle_embedding = nn.Embedding(CYCLE_LENGTH, config.cycle_embedding_dim)
        self.phase_embedding = nn.Embedding(GAME_PHASE_COUNT, config.phase_embedding_dim)
        self.board_size_embedding = nn.Embedding(len(BOARD_SIZES), config.board_size_embedding_dim)
        categorical_channels = config.cycle_embedding_dim + config.phase_embedding_dim + config.board_size_embedding_dim
        input_channels = len(SPATIAL_FEATURE_NAMES) + categorical_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(self.BLOCK_COUNT))
        self.output_channels = channels
        self.global_output_channels = channels * 2 + categorical_channels
        self.register_buffer(
            "spatial_indices",
            torch.tensor([FEATURE_INDEX[name] for name in SPATIAL_FEATURE_NAMES]),
            persistent=False,
        )

    @staticmethod
    def _category_index(inputs: Tensor, name: str, category_count: int) -> Tensor:
        values = inputs[:, FEATURE_INDEX[name]].amax(dim=(-2, -1))
        return values.round().long().clamp_(0, category_count - 1)

    def forward_features(self, inputs: Tensor) -> EncoderFeatures:
        mask = inputs[:, FEATURE_INDEX["board_mask"] : FEATURE_INDEX["board_mask"] + 1]
        spatial_inputs = inputs.index_select(1, self.spatial_indices)
        categorical = torch.cat(
            (
                self.cycle_embedding(self._category_index(inputs, "day_night_cycle", CYCLE_LENGTH)),
                self.phase_embedding(self._category_index(inputs, "game_phase", GAME_PHASE_COUNT)),
                self.board_size_embedding(self._category_index(inputs, "board_size", len(BOARD_SIZES))),
            ),
            dim=1,
        )
        categorical_map = categorical[:, :, None, None].expand(-1, -1, inputs.shape[-2], inputs.shape[-1])
        spatial = self.stem(torch.cat((spatial_inputs, categorical_map), dim=1) * mask) * mask
        for block in self.blocks:
            spatial = block(spatial, mask)

        valid_count = mask.sum(dim=(-2, -1)).clamp_min(1)
        pooled_average = (spatial * mask).sum(dim=(-2, -1)) / valid_count
        pooled_maximum = spatial.masked_fill(mask <= 0, torch.finfo(spatial.dtype).min).amax(dim=(-2, -1))
        global_features = torch.cat((pooled_average, pooled_maximum, categorical), dim=1)
        return EncoderFeatures(spatial=spatial, global_features=global_features)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_features(inputs).spatial


class ResNet17x32SpatialEncoder(ResNet17SpatialEncoder):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, channels=32)


class ResNet17x48SpatialEncoder(ResNet17SpatialEncoder):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config, channels=48)


class Transformer16SpatialEncoder(nn.Module):
    STEM_CHANNELS = 96

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.input_channels != len(FEATURE_NAMES):
            msg = f"Expected {len(FEATURE_NAMES)} input channels, got {config.input_channels}"
            raise ValueError(msg)
        channels = config.transformer_dim
        self.cycle_embedding = nn.Embedding(CYCLE_LENGTH, config.cycle_embedding_dim)
        self.phase_embedding = nn.Embedding(GAME_PHASE_COUNT, config.phase_embedding_dim)
        self.board_size_embedding = nn.Embedding(len(BOARD_SIZES), config.board_size_embedding_dim)
        categorical_channels = config.cycle_embedding_dim + config.phase_embedding_dim + config.board_size_embedding_dim
        input_channels = len(SPATIAL_FEATURE_NAMES) + categorical_channels
        self.stem = _EncoderStage(input_channels, self.STEM_CHANNELS)
        self.downsample = nn.Sequential(
            nn.Conv2d(self.STEM_CHANNELS, channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ffn_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.transformer16_layers,
            norm=nn.LayerNorm(channels),
            enable_nested_tensor=False,
        )
        self.decoder = _EncoderStage(channels + self.STEM_CHANNELS, config.feature_channels)
        self.output_channels = config.feature_channels
        self.global_output_channels = config.feature_channels * 2 + categorical_channels
        self.register_buffer(
            "spatial_indices",
            torch.tensor([FEATURE_INDEX[name] for name in SPATIAL_FEATURE_NAMES]),
            persistent=False,
        )

    @staticmethod
    def _category_index(inputs: Tensor, name: str, category_count: int) -> Tensor:
        values = inputs[:, FEATURE_INDEX[name]].amax(dim=(-2, -1))
        return values.round().long().clamp_(0, category_count - 1)

    def forward_features(self, inputs: Tensor) -> EncoderFeatures:
        mask0 = inputs[:, FEATURE_INDEX["board_mask"] : FEATURE_INDEX["board_mask"] + 1]
        spatial_inputs = inputs.index_select(1, self.spatial_indices)
        categorical = torch.cat(
            (
                self.cycle_embedding(self._category_index(inputs, "day_night_cycle", CYCLE_LENGTH)),
                self.phase_embedding(self._category_index(inputs, "game_phase", GAME_PHASE_COUNT)),
                self.board_size_embedding(self._category_index(inputs, "board_size", len(BOARD_SIZES))),
            ),
            dim=1,
        )
        categorical_map = categorical[:, :, None, None].expand(-1, -1, inputs.shape[-2], inputs.shape[-1])
        stem_inputs = torch.cat((spatial_inputs, categorical_map), dim=1) * mask0
        skip = self.stem(stem_inputs, mask0)

        mask1 = nn_functional.max_pool2d(mask0, kernel_size=2, stride=2)
        tokens_2d = self.downsample(skip) * mask1
        batch_size, channels, height, width = tokens_2d.shape
        tokens = tokens_2d.flatten(2).transpose(1, 2)
        padding_mask = mask1.flatten(1) <= 0
        tokens = self.transformer(tokens, src_key_padding_mask=padding_mask)
        tokens_2d = tokens.transpose(1, 2).reshape(batch_size, channels, height, width) * mask1

        upsampled = nn_functional.interpolate(
            tokens_2d,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        spatial = self.decoder(torch.cat((upsampled, skip), dim=1), mask0)
        valid_count = mask0.sum(dim=(-2, -1)).clamp_min(1)
        pooled_average = (spatial * mask0).sum(dim=(-2, -1)) / valid_count
        pooled_maximum = spatial.masked_fill(mask0 <= 0, torch.finfo(spatial.dtype).min).amax(dim=(-2, -1))
        global_features = torch.cat((pooled_average, pooled_maximum, categorical), dim=1)
        return EncoderFeatures(spatial=spatial, global_features=global_features)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_features(inputs).spatial


class AxisAttention(nn.Module):
    def __init__(self, channels: int, heads: int, dropout: float, maximum_length: int = 32) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = 1 / sqrt(self.head_dim)
        self.qkv = nn.Linear(channels, channels * 3)
        self.projection = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(maximum_length)
        self.register_buffer(
            "distance_indices",
            (positions[:, None] - positions[None, :]).abs(),
            persistent=False,
        )

    def forward(self, inputs: Tensor, valid_mask: Tensor, distance_bias: Tensor) -> Tensor:
        batch_size, group_count, length, channels = inputs.shape
        flattened = inputs.reshape(batch_size * group_count, length, channels)
        flat_mask = valid_mask.reshape(batch_size * group_count, length).bool()
        qkv = self.qkv(flattened).reshape(
            batch_size * group_count,
            length,
            3,
            self.heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        indices = self.distance_indices[:length, :length]
        empty_groups = ~flat_mask.any(dim=1)
        first_position = torch.arange(length, device=inputs.device) == 0
        safe_mask = flat_mask | (empty_groups[:, None] & first_position[None])
        attention_bias = distance_bias[:, indices].unsqueeze(0).to(dtype=query.dtype)
        attention_bias = attention_bias.expand(batch_size * group_count, -1, -1, -1)
        attention_bias = attention_bias.masked_fill(~safe_mask[:, None, None, :], float("-inf"))
        attended = nn_functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=self.dropout.p if self.training else 0.0,
            scale=self.scale,
        )
        attended = attended.transpose(1, 2).reshape(batch_size * group_count, length, channels)
        attended = self.projection(attended).reshape(batch_size, group_count, length, channels)
        return attended * valid_mask[..., None]


class AxialTransformerBlock(nn.Module):
    def __init__(self, channels: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.position = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.row_norm = nn.LayerNorm(channels)
        self.column_norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.row_attention = AxisAttention(channels, heads, dropout)
        self.column_attention = AxisAttention(channels, heads, dropout)
        self.distance_bias = nn.Parameter(torch.zeros(heads, 32))
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, channels),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        output = (inputs + self.position(inputs)) * mask
        cell_mask = mask[:, 0].bool()

        row_inputs = output.permute(0, 2, 3, 1)
        row_update = self.row_attention(self.row_norm(row_inputs), cell_mask, self.distance_bias)
        output = (output + row_update.permute(0, 3, 1, 2)) * mask

        column_inputs = output.permute(0, 3, 2, 1)
        column_mask = cell_mask.transpose(1, 2)
        column_update = self.column_attention(
            self.column_norm(column_inputs),
            column_mask,
            self.distance_bias,
        )
        output = (output + column_update.permute(0, 3, 2, 1)) * mask

        cell_inputs = output.permute(0, 2, 3, 1)
        ffn_update = self.ffn(self.ffn_norm(cell_inputs))
        return (output + ffn_update.permute(0, 3, 1, 2)) * mask


class Axial32SpatialEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.input_channels != len(FEATURE_NAMES):
            msg = f"Expected {len(FEATURE_NAMES)} input channels, got {config.input_channels}"
            raise ValueError(msg)
        compact = config.encoder_type == "axial32_4m5"
        channels = config.axial32_4m5_dim if compact else config.transformer_dim
        ffn_dim = config.axial32_4m5_ffn_dim if compact else config.transformer_ffn_dim
        layer_count = config.axial32_4m5_layers if compact else config.axial32_layers
        self.cycle_embedding = nn.Embedding(CYCLE_LENGTH, config.cycle_embedding_dim)
        self.phase_embedding = nn.Embedding(GAME_PHASE_COUNT, config.phase_embedding_dim)
        self.board_size_embedding = nn.Embedding(len(BOARD_SIZES), config.board_size_embedding_dim)
        categorical_channels = config.cycle_embedding_dim + config.phase_embedding_dim + config.board_size_embedding_dim
        input_channels = len(SPATIAL_FEATURE_NAMES) + categorical_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            AxialTransformerBlock(
                channels,
                config.transformer_heads,
                ffn_dim,
                config.transformer_dropout,
            )
            for _ in range(layer_count)
        )
        self.output = _EncoderStage(channels, config.feature_channels)
        self.output_channels = config.feature_channels
        self.global_output_channels = config.feature_channels * 2 + categorical_channels
        self.register_buffer(
            "spatial_indices",
            torch.tensor([FEATURE_INDEX[name] for name in SPATIAL_FEATURE_NAMES]),
            persistent=False,
        )

    @staticmethod
    def _category_index(inputs: Tensor, name: str, category_count: int) -> Tensor:
        values = inputs[:, FEATURE_INDEX[name]].amax(dim=(-2, -1))
        return values.round().long().clamp_(0, category_count - 1)

    def forward_features(self, inputs: Tensor) -> EncoderFeatures:
        mask = inputs[:, FEATURE_INDEX["board_mask"] : FEATURE_INDEX["board_mask"] + 1]
        spatial_inputs = inputs.index_select(1, self.spatial_indices)
        categorical = torch.cat(
            (
                self.cycle_embedding(self._category_index(inputs, "day_night_cycle", CYCLE_LENGTH)),
                self.phase_embedding(self._category_index(inputs, "game_phase", GAME_PHASE_COUNT)),
                self.board_size_embedding(self._category_index(inputs, "board_size", len(BOARD_SIZES))),
            ),
            dim=1,
        )
        categorical_map = categorical[:, :, None, None].expand(-1, -1, inputs.shape[-2], inputs.shape[-1])
        spatial = self.stem(torch.cat((spatial_inputs, categorical_map), dim=1) * mask) * mask
        for block in self.blocks:
            spatial = block(spatial, mask)
        spatial = self.output(spatial, mask)

        valid_count = mask.sum(dim=(-2, -1)).clamp_min(1)
        pooled_average = (spatial * mask).sum(dim=(-2, -1)) / valid_count
        pooled_maximum = spatial.masked_fill(mask <= 0, torch.finfo(spatial.dtype).min).amax(dim=(-2, -1))
        global_features = torch.cat((pooled_average, pooled_maximum, categorical), dim=1)
        return EncoderFeatures(spatial=spatial, global_features=global_features)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_features(inputs).spatial


class ActionPredictionHead(nn.Module):
    def __init__(self, feature_channels: int, action_count: int) -> None:
        super().__init__()
        hidden_channels = max(32, feature_channels // 2)
        self.network = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, action_count, kernel_size=1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class LuxBehaviorCloningModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        encoder_types = {
            "unet": LuxSpatialEncoder,
            "resnet17x32": ResNet17x32SpatialEncoder,
            "resnet17x48": ResNet17x48SpatialEncoder,
            "resattn8": ResAttnUNet8SpatialEncoder,
            "transformer16": Transformer16SpatialEncoder,
            "axial32": Axial32SpatialEncoder,
            "axial32_4m5": Axial32SpatialEncoder,
        }
        self.encoder = encoder_types[self.config.encoder_type](self.config)
        action_schema = (
            ACTION_SCHEMA if self.config.policy_schema == POLICY_SCHEMA_FACTORIZED else FIRST_PLACE_ACTION_SCHEMA
        )
        self.heads = nn.ModuleDict(
            {
                name: ActionPredictionHead(self.encoder.output_channels, len(actions))
                for name, actions in action_schema.items()
            }
        )

    def encode(self, observation: Tensor) -> Tensor:
        return self.encoder(observation)

    def encode_with_global(self, observation: Tensor) -> EncoderFeatures:
        return self.encoder.forward_features(observation)

    def forward(
        self,
        observation: Tensor,
        *,
        return_features: bool = False,
    ) -> dict[str, Tensor]:
        encoded = self.encode_with_global(observation)
        output = {name: head(encoded.spatial) for name, head in self.heads.items()}
        if return_features:
            output["features"] = encoded.spatial
            output["global_features"] = encoded.global_features
        return output


def _safe_cross_entropy(
    logits: Tensor,
    target: Tensor,
    sample_weight: Tensor,
    class_weight: Tensor | None,
) -> Tensor:
    valid = target != IGNORE_INDEX
    if not torch.any(valid):
        # Keep a gradient connection without reducing the entire logits tensor.
        # A large but finite tensor can overflow during sum(), making inf * 0
        # become NaN on batches that contain no target for this factorized head.
        return logits.reshape(-1)[0] * 0
    losses = nn_functional.cross_entropy(
        logits,
        target,
        weight=class_weight,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    weight_shape = (sample_weight.shape[0],) + (1,) * (losses.ndim - 1)
    weights = sample_weight.reshape(weight_shape).expand_as(losses)
    weighted_valid = weights * valid
    return (losses * weighted_valid).sum() / weighted_valid.sum().clamp_min(1)


def _entity_logits(logits: Tensor, positions: Tensor) -> Tensor:
    batch_size = logits.shape[0]
    safe_positions = positions.clamp_min(0)
    batch_indices = torch.arange(batch_size, device=logits.device)[:, None]
    gathered = logits[
        batch_indices,
        :,
        safe_positions[:, :, 0],
        safe_positions[:, :, 1],
    ]
    return gathered.permute(0, 2, 1)


def behavior_cloning_loss(
    output: Mapping[str, Tensor],
    batch: Mapping[str, Tensor],
    class_weights: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    sample_weight = batch["sample_weight"]
    losses = {}
    for name in TARGET_NAMES:
        weight = None if class_weights is None else class_weights.get(name)
        entity = name.split("_", maxsplit=1)[0]
        logits = _entity_logits(output[name], batch[f"{entity}_positions"])
        legal_mask = batch[f"{name}{LEGAL_MASK_SUFFIX}"].permute(0, 2, 1)
        logits = apply_legal_action_mask(logits, legal_mask, action_dim=1)
        losses[f"{name}_loss"] = _safe_cross_entropy(logits, batch[name], sample_weight, weight)
    total = torch.stack(tuple(losses.values())).sum()
    return {"loss": total, **losses}


def compute_metrics(output: Mapping[str, Tensor], batch: Mapping[str, Tensor]) -> dict[str, tuple[Tensor, Tensor]]:
    metrics = {}
    for name in TARGET_NAMES:
        target = batch[name]
        valid = target != IGNORE_INDEX
        entity = name.split("_", maxsplit=1)[0]
        logits = _entity_logits(output[name], batch[f"{entity}_positions"])
        legal_mask = batch[f"{name}{LEGAL_MASK_SUFFIX}"].permute(0, 2, 1)
        logits = apply_legal_action_mask(logits, legal_mask, action_dim=1)
        predictions = logits.argmax(dim=1)
        metrics[f"{name}_accuracy"] = (
            ((predictions == target) & valid).sum(),
            valid.sum(),
        )
    for prefix in ("worker", "cart"):
        target = batch[f"{prefix}_type"]
        valid = (target != IGNORE_INDEX) & (target != 0)
        logits = _entity_logits(output[f"{prefix}_type"], batch[f"{prefix}_positions"])
        logits = apply_legal_action_mask(
            logits,
            batch[f"{prefix}_type{LEGAL_MASK_SUFFIX}"].permute(0, 2, 1),
            action_dim=1,
        )
        predictions = logits.argmax(dim=1)
        metrics[f"{prefix}_active_accuracy"] = (
            ((predictions == target) & valid).sum(),
            valid.sum(),
        )
    city_target = batch["city"]
    city_valid = (city_target != IGNORE_INDEX) & (city_target != 0)
    city_logits = _entity_logits(output["city"], batch["city_positions"])
    city_legal = batch[f"city{LEGAL_MASK_SUFFIX}"].permute(0, 2, 1)
    city_logits = apply_legal_action_mask(city_logits, city_legal, action_dim=1)
    city_predictions = city_logits.argmax(dim=1)
    metrics["city_active_accuracy"] = (
        ((city_predictions == city_target) & city_valid).sum(),
        city_valid.sum(),
    )
    return metrics


def compute_confusion_matrices(output: Mapping[str, Tensor], batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
    matrices = {}
    for name, actions in ACTION_SCHEMA.items():
        target = batch[name]
        valid = target != IGNORE_INDEX
        entity = name.split("_", maxsplit=1)[0]
        logits = _entity_logits(output[name], batch[f"{entity}_positions"])
        legal_mask = batch[f"{name}{LEGAL_MASK_SUFFIX}"].permute(0, 2, 1)
        predictions = apply_legal_action_mask(logits, legal_mask, action_dim=1).argmax(dim=1)
        action_count = len(actions)
        encoded_pairs = target[valid] * action_count + predictions[valid]
        matrices[name] = torch.bincount(
            encoded_pairs,
            minlength=action_count * action_count,
        ).reshape(action_count, action_count)
    return matrices


def make_class_weights(
    counts: Mapping[str, Tensor],
    device: torch.device,
    exponent: float = 0.5,
) -> dict[str, Tensor]:
    result = {}
    for name, count_values in counts.items():
        values = count_values.to(device=device, dtype=torch.float32)
        nonzero = values > 0
        weights = torch.ones_like(values)
        if torch.any(nonzero):
            weights[nonzero] = values[nonzero].pow(-exponent)
            weights[nonzero] /= weights[nonzero].mean()
        result[name] = weights
    return result


def save_bc_checkpoint(  # noqa: PLR0913
    path: Path,
    model: LuxBehaviorCloningModel,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    metrics: Mapping[str, float],
    split: Mapping[str, object],
    *,
    class_counts: Mapping[str, Tensor] | None = None,
    class_statistics_signature: str | None = None,
    scaler: torch.amp.GradScaler | None = None,
    extra_metadata: Mapping[str, object] | None = None,
) -> None:
    action_schema = (
        ACTION_SCHEMA if model.config.policy_schema == POLICY_SCHEMA_FACTORIZED else FIRST_PLACE_ACTION_SCHEMA
    )
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_schema": action_schema,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "epoch": epoch,
        "metrics": dict(metrics),
        "split": dict(split),
        "class_counts": (
            None if class_counts is None else {name: values.detach().cpu() for name, values in class_counts.items()}
        ),
        "class_statistics_signature": class_statistics_signature,
    }
    if extra_metadata:
        checkpoint.update(dict(extra_metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_bc_checkpoint(
    path: str,
    device: str = "cpu",
    load_optimizer: torch.optim.Optimizer | None = None,
) -> tuple[LuxBehaviorCloningModel, Mapping[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported behavior-cloning checkpoint schema")
    if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Checkpoint feature schema version does not match this package")
    if tuple(checkpoint.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature schema does not match this package")
    config = ModelConfig(**checkpoint["model_config"])
    expected_action_schema = (
        ACTION_SCHEMA if config.policy_schema == POLICY_SCHEMA_FACTORIZED else FIRST_PLACE_ACTION_SCHEMA
    )
    if checkpoint.get("action_schema") != expected_action_schema:
        raise ValueError("Checkpoint action schema does not match this package")
    nonfinite_parameters = [
        name
        for name, value in checkpoint["model"].items()
        if torch.is_floating_point(value) and not torch.isfinite(value).all()
    ]
    if nonfinite_parameters:
        preview = ", ".join(nonfinite_parameters[:3])
        message = f"Checkpoint contains non-finite model parameters: {preview}"
        raise ValueError(message)
    model = LuxBehaviorCloningModel(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    if load_optimizer is not None and checkpoint.get("optimizer") is not None:
        load_optimizer.load_state_dict(checkpoint["optimizer"])
    return model, checkpoint
