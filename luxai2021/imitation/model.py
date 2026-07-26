from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import functional as nn_functional

from luxai2021.imitation.actions import ACTION_SCHEMA, TARGET_NAMES
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


class FReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.maximum(inputs, self.norm(self.depthwise(inputs)))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        hidden_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.expand = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        scale = self.pool(inputs)
        scale = nn_functional.relu(self.reduce(scale), inplace=True)
        return inputs * torch.sigmoid(self.expand(scale))


class DurrettResidualBlock(nn.Module):
    def __init__(self, channels: int, se_reduction: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.se = SqueezeExcitation(channels, se_reduction)
        self.activation = FReLU(channels)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        residual = self.se(self.norm(self.conv(inputs))) * mask
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
    durrett_layers: int = 18
    transformer_layers: int = 1
    transformer_heads: int = 1
    se_reduction: int = 16
    source_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_ids", tuple(int(source_id) for source_id in self.source_ids))
        if self.encoder_type not in {"unet", "durrett"}:
            msg = f"Unsupported encoder type: {self.encoder_type}"
            raise ValueError(msg)
        if len(set(self.source_ids)) != len(self.source_ids):
            raise ValueError("source_ids must be unique")
        if self.feature_channels % self.transformer_heads:
            raise ValueError("feature_channels must be divisible by transformer_heads")


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


class DurrettSpatialEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.input_channels != len(FEATURE_NAMES):
            msg = f"Expected {len(FEATURE_NAMES)} input channels, got {config.input_channels}"
            raise ValueError(msg)
        channels = config.feature_channels
        self.cycle_embedding = nn.Embedding(CYCLE_LENGTH, config.cycle_embedding_dim)
        self.phase_embedding = nn.Embedding(GAME_PHASE_COUNT, config.phase_embedding_dim)
        self.board_size_embedding = nn.Embedding(len(BOARD_SIZES), config.board_size_embedding_dim)
        categorical_channels = config.cycle_embedding_dim + config.phase_embedding_dim + config.board_size_embedding_dim
        input_channels = len(SPATIAL_FEATURE_NAMES) + categorical_channels
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(
            DurrettResidualBlock(channels, config.se_reduction) for _ in range(config.durrett_layers)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=config.transformer_heads,
            dim_feedforward=channels,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.transformer_layers,
            enable_nested_tensor=False,
        )
        self.position_embedding = nn.Parameter(torch.randn(1, 32 * 32, channels) * 0.02)
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
        board_mask = inputs[:, FEATURE_INDEX["board_mask"] : FEATURE_INDEX["board_mask"] + 1]
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
        spatial = self.stem(torch.cat((spatial_inputs, categorical_map), dim=1) * board_mask) * board_mask
        for block in self.blocks:
            spatial = block(spatial, board_mask)

        batch_size, channels, height, width = spatial.shape
        flattened = spatial.flatten(2).transpose(1, 2)
        padding_mask = board_mask.flatten(2).squeeze(1) <= 0
        attended = self.transformer(
            flattened + self.position_embedding[:, : height * width],
            src_key_padding_mask=padding_mask,
        )
        spatial = (flattened + attended).transpose(1, 2).reshape(batch_size, channels, height, width) * board_mask

        valid_count = board_mask.sum(dim=(-2, -1)).clamp_min(1)
        pooled_average = spatial.sum(dim=(-2, -1)) / valid_count
        pooled_maximum = spatial.masked_fill(board_mask <= 0, torch.finfo(spatial.dtype).min).amax(dim=(-2, -1))
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


class MultiSourceActionPredictionHead(nn.Module):
    def __init__(self, feature_channels: int, action_count: int, source_count: int) -> None:
        super().__init__()
        hidden_channels = max(32, feature_channels // 2)
        self.projection = nn.Sequential(
            nn.Conv2d(feature_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_channels), hidden_channels),
            nn.SiLU(),
        )
        self.classifiers = nn.ModuleList(
            nn.Conv2d(hidden_channels, action_count, kernel_size=1) for _ in range(source_count)
        )

    def forward(self, features: Tensor, source_index: Tensor | int | None) -> Tensor:
        projected = self.projection(features)
        if isinstance(source_index, int):
            return self.classifiers[source_index](projected)
        if source_index is None:
            if len(self.classifiers) != 1:
                raise ValueError("source_index is required for a multi-source model")
            return self.classifiers[0](projected)
        source_index = source_index.to(device=features.device, dtype=torch.long).reshape(-1)
        if source_index.shape[0] != features.shape[0]:
            raise ValueError("source_index batch dimension does not match observation")
        if torch.any((source_index < 0) | (source_index >= len(self.classifiers))):
            raise ValueError("source_index is outside the configured source range")
        if torch.all(source_index == source_index[0]):
            return self.classifiers[int(source_index[0].item())](projected)
        all_logits = torch.stack([classifier(projected) for classifier in self.classifiers], dim=1)
        gather_index = source_index[:, None, None, None, None].expand(
            -1,
            1,
            all_logits.shape[2],
            all_logits.shape[3],
            all_logits.shape[4],
        )
        return all_logits.gather(1, gather_index).squeeze(1)


class LuxBehaviorCloningModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = (
            DurrettSpatialEncoder(self.config)
            if self.config.encoder_type == "durrett"
            else LuxSpatialEncoder(self.config)
        )
        if self.config.source_ids:
            self.heads = nn.ModuleDict(
                {
                    name: MultiSourceActionPredictionHead(
                        self.encoder.output_channels,
                        len(actions),
                        len(self.config.source_ids),
                    )
                    for name, actions in ACTION_SCHEMA.items()
                }
            )
        else:
            self.heads = nn.ModuleDict(
                {
                    name: ActionPredictionHead(self.encoder.output_channels, len(actions))
                    for name, actions in ACTION_SCHEMA.items()
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
        source_index: Tensor | int | None = None,
        return_features: bool = False,
    ) -> dict[str, Tensor]:
        encoded = self.encode_with_global(observation)
        if self.config.source_ids:
            output = {name: head(encoded.spatial, source_index) for name, head in self.heads.items()}
        else:
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
        return logits[:, 0].sum() * 0
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
    stay_weight: float | None = None,
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
    if stay_weight is not None:
        result["worker_type"][0] *= stay_weight
        result["cart_type"][0] *= stay_weight
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
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    training_profile: str = "baseline",
    source_catalog: tuple[Mapping[str, object], ...] = (),
    default_source_id: int | None = None,
) -> None:
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": FEATURE_NAMES,
        "action_schema": ACTION_SCHEMA,
        "model_config": asdict(model.config),
        "model": model.state_dict(),
        "encoder": model.encoder.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "metrics": dict(metrics),
        "split": dict(split),
        "class_counts": (
            None if class_counts is None else {name: values.detach().cpu() for name, values in class_counts.items()}
        ),
        "class_statistics_signature": class_statistics_signature,
        "training_profile": training_profile,
        "source_catalog": tuple(dict(source) for source in source_catalog),
        "default_source_id": default_source_id,
    }
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
    if checkpoint.get("action_schema") != ACTION_SCHEMA:
        raise ValueError("Checkpoint action schema does not match this package")
    model = LuxBehaviorCloningModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    if load_optimizer is not None and checkpoint.get("optimizer") is not None:
        load_optimizer.load_state_dict(checkpoint["optimizer"])
    return model, checkpoint
