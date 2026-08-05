# ruff: noqa: ANN001, ANN201, ANN202, PLR2004, S101

import argparse
import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from examples.compare_bc_architectures import build_comparison
from examples.evaluate_bc_checkpoints import (
    aggregate_match_results,
    expected_match_game_count,
    match_pair_orientations,
    parse_checkpoint,
    parse_match_workers,
    resolve_match_workers,
    select_evaluation_winner,
    shard_match_seeds,
    sort_match_games,
)
from examples.train_bc import main as train_main
from examples.train_bc import resolve_amp_dtype, resolve_compile, run_epoch
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from luxai2021.imitation.actions import CITY_ACTIONS, DIRECTIONS, RESOURCES, WORKER_ACTIONS
from luxai2021.imitation.agent import BehaviorCloningAgent
from luxai2021.imitation.class_stats import (
    checkpoint_class_statistics,
    class_statistics_signature,
    load_class_statistics,
    save_class_statistics,
)
from luxai2021.imitation.data import (
    IGNORE_INDEX,
    LuxReplayDataset,
    ReplayBatchSampler,
    augment_sample,
    build_targets,
    class_counts,
    prepare_replay_cache,
)
from luxai2021.imitation.masking import (
    apply_legal_action_mask,
    build_legal_masks,
)
from luxai2021.imitation.model import (
    POLICY_SCHEMA_FIRST_PLACE_FLAT,
    AxialTransformerBlock,
    AxisAttention,
    GlobalSpatialAttentionBlock,
    LuxBehaviorCloningModel,
    ModelConfig,
    _safe_cross_entropy,
    behavior_cloning_loss,
    compute_confusion_matrices,
    load_bc_checkpoint,
    save_bc_checkpoint,
)
from luxai2021.imitation.schema import (
    BOARD_SIZES,
    CYCLE_LENGTH,
    FEATURE_INDEX,
    FEATURE_NAMES,
    GAME_PHASE_COUNT,
    encode_snapshot,
    snapshot_from_game,
    snapshot_from_updates,
)


def _updates():
    return [
        "rp 0 50",
        "rp 1 0",
        "r wood 0 0 400",
        "u 0 0 u_1 1 1 0 10 0 0",
        "u 1 0 u_2 2 1 0 0 0 0",
        "u 0 1 u_3 14 14 0 0 0 0",
        "c 0 c_1 230 23",
        "ct 0 c_1 1 2 0",
        "ccd 1 1 3",
        "D_DONE",
    ]


def test_snapshot_encoding_is_centered_and_team_relative():
    snapshot = snapshot_from_updates(_updates(), width=16, height=16, turn=30)
    own = encode_snapshot(snapshot, team=0)
    enemy = encode_snapshot(snapshot, team=1)

    assert own.shape == (len(FEATURE_NAMES), 32, 32)
    assert own[FEATURE_INDEX["board_mask"], 8:24, 8:24].all()
    assert not own[FEATURE_INDEX["board_mask"], :8].any()
    assert own[FEATURE_INDEX["own_worker"], 9, 9] == 1
    assert own[FEATURE_INDEX["own_worker_count"], 9, 9] == 1
    assert enemy[FEATURE_INDEX["enemy_worker"], 9, 9] == 1
    assert own[FEATURE_INDEX["own_coal_researched"], 8, 8] == 1
    assert own[FEATURE_INDEX["is_night"], 8, 8] == 1
    assert np.isclose(own[FEATURE_INDEX["wood"], 8, 8], 0.5)
    assert own[FEATURE_INDEX["day_night_cycle"], 8, 8] == 30
    assert own[FEATURE_INDEX["game_phase"], 8, 8] == 0
    assert own[FEATURE_INDEX["board_size"], 8, 8] == BOARD_SIZES.index(16)


def test_hybrid_features_encode_stacks_full_cargo_and_categories():
    updates = [
        "u 0 0 u_1 1 1 3 100 0 0",
        "u 0 0 u_2 1 1 0 0 0 0",
        "D_DONE",
    ]
    snapshot = snapshot_from_updates(updates, width=16, height=16, turn=85)
    features = encode_snapshot(snapshot, team=0)
    x, y = snapshot.padded_position(1, 1)

    assert len(FEATURE_NAMES) == 55
    assert features[FEATURE_INDEX["own_worker_count"], y, x] == 2
    assert features[FEATURE_INDEX["own_worker_cooldown"], y, x] == 1
    assert features[FEATURE_INDEX["own_worker_wood"], y, x] == 1
    assert features[FEATURE_INDEX["own_worker_cargo_full"], y, x] == 1
    assert features[FEATURE_INDEX["day_night_cycle"], y, x] == 85 % CYCLE_LENGTH
    assert features[FEATURE_INDEX["game_phase"], y, x] == min(85 // CYCLE_LENGTH, GAME_PHASE_COUNT - 1)


def test_targets_include_implicit_stay_and_factorized_transfer():
    snapshot = snapshot_from_updates(_updates(), width=16, height=16, turn=0)
    targets = build_targets(snapshot, 0, ["t u_1 u_2 wood 10"])

    assert targets["worker_type"][0] == 3
    assert targets["worker_transfer_dir"][0] == 1
    assert targets["worker_resource"][0] == 0
    assert targets["cart_type"][0] == 0
    assert targets["city"][0] == 0


def test_augmentation_rotates_direction_labels():
    snapshot = snapshot_from_updates(_updates(), width=16, height=16, turn=0)
    observation = encode_snapshot(snapshot, 0)
    targets = build_targets(snapshot, 0, ["m u_1 n"])
    _, rotated = augment_sample(observation, targets, rotations=1, horizontal_flip=False)

    valid = rotated["worker_move"] != IGNORE_INDEX
    assert rotated["worker_move"][valid].item() == 3


def test_legal_masks_follow_reference_viability_rules():
    updates = [
        "rp 0 200",
        "rp 1 0",
        "u 0 0 u_worker 0 0 0 100 0 0",
        "u 1 0 u_blocker 1 0 2 0 0 0",
        "u 1 0 u_cart 2 2 0 0 0 0",
        "c 1 c_enemy 100 23",
        "ct 1 c_enemy 0 1 0",
        "c 0 c_own 100 23",
        "ct 0 c_own 3 3 0",
        "ccd 0 0 1",
        "ccd 2 2 1",
        "D_DONE",
    ]
    snapshot = snapshot_from_updates(updates, width=16, height=16, turn=0)
    masks = build_legal_masks(snapshot, team=0)

    worker_type = masks["worker_type_legal_mask"][0]
    assert worker_type[WORKER_ACTIONS.index("stay")]
    assert not worker_type[WORKER_ACTIONS.index("move")]
    assert worker_type[WORKER_ACTIONS.index("build_city")]
    assert worker_type[WORKER_ACTIONS.index("transfer")]
    assert not masks["worker_move_legal_mask"][0].any()
    assert masks["worker_transfer_dir_legal_mask"][0, DIRECTIONS.index("e")]
    assert masks["worker_resource_legal_mask"][0, RESOURCES.index("wood")]
    assert masks["cart_type_legal_mask"][0].tolist() == [
        True,
        True,
        False,
        False,
    ]

    city = masks["city_legal_mask"][0]
    assert city[CITY_ACTIONS.index("no_action")]
    assert not city[CITY_ACTIONS.index("build_worker")]
    assert not city[CITY_ACTIONS.index("build_cart")]
    assert not city[CITY_ACTIONS.index("research")]


def test_illegal_replay_command_becomes_noop_and_masked_logits_are_finite():
    updates = [
        "u 0 0 u_worker 0 0 0 0 0 0",
        "u 1 0 u_blocker 1 0 2 0 0 0",
        "D_DONE",
    ]
    snapshot = snapshot_from_updates(updates, width=16, height=16, turn=0)
    targets = build_targets(snapshot, team=0, actions=["m u_worker e"])
    assert targets["worker_type"][0] == WORKER_ACTIONS.index("stay")
    assert targets["worker_move"][0] == IGNORE_INDEX

    logits = torch.tensor([[0.0, 100.0, 2.0, 50.0]])
    legal = torch.tensor([[True, False, True, False]])
    masked = apply_legal_action_mask(logits, legal)
    assert masked.argmax(dim=-1).item() == 2
    assert torch.isfinite(torch.log_softmax(masked, dim=-1)).all()


def test_empty_target_loss_does_not_overflow_when_logits_are_large():
    logits = torch.full((2, 1024, 4), 1e38, requires_grad=True)
    target = torch.full((2, 1024), IGNORE_INDEX)
    sample_weight = torch.ones(2)

    loss = _safe_cross_entropy(logits, target, sample_weight, None)
    loss.backward()

    assert loss.item() == 0
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad.any()


def test_augmentation_rotates_direction_masks():
    updates = ["u 0 0 u_worker 1 0 0 0 0 0", "D_DONE"]
    snapshot = snapshot_from_updates(updates, width=16, height=16, turn=0)
    observation = encode_snapshot(snapshot, 0)
    targets = build_targets(snapshot, 0, [])
    _, rotated = augment_sample(observation, targets, rotations=1, horizontal_flip=False)

    # A north edge becomes the west edge after np.rot90(..., 1).
    assert not rotated["worker_move_legal_mask"][0, DIRECTIONS.index("w")]
    assert rotated["worker_move_legal_mask"][0, DIRECTIONS.index("e")]


def test_stacked_units_keep_separate_labels():
    updates = [*_updates(), "u 0 0 u_4 1 1 0 0 0 0"]
    snapshot = snapshot_from_updates(updates, width=16, height=16, turn=0)
    targets = build_targets(snapshot, 0, ["m u_1 n", "m u_4 e"])

    assert np.array_equal(targets["worker_positions"][0], targets["worker_positions"][1])
    assert np.array_equal(targets["worker_move"][:2], [0, 1])


def test_model_loss_and_checkpoint_round_trip(tmp_path):
    replay = {
        "rewards": [1, 0],
        "steps": [
            [
                {"observation": {"width": 16, "height": 16, "updates": _updates()}},
                {"observation": {}},
            ],
            [{"action": ["m u_1 n"], "step": 1}, {"action": ["m u_3 s"], "step": 1}],
        ],
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    dataset = LuxReplayDataset([replay_path])
    assert len(dataset) == 1
    assert dataset[0]["sample_weight"] == 1
    all_players = LuxReplayDataset([replay_path], team_selection="all")
    assert len(all_players) == 2
    assert [float(all_players[index]["sample_weight"]) for index in range(2)] == [1.5, 1.0]
    assert list(ReplayBatchSampler(all_players, batch_size=1, shuffle=False)) == [[0], [1]]
    batch = next(iter(DataLoader(dataset, batch_size=2)))
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=8, feature_channels=16))
    output = model(batch["observation"])
    losses = behavior_cloning_loss(output, batch)
    confusion_matrices = compute_confusion_matrices(output, batch)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert confusion_matrices["worker_type"].shape == (4, 4)
    assert confusion_matrices["worker_type"].sum() == (batch["worker_type"] != IGNORE_INDEX).sum()
    counts = class_counts(dataset)
    statistics_signature = class_statistics_signature(
        [replay_path],
        team_selection="winner",
        max_turns=0,
    )
    statistics_path = tmp_path / "class_statistics.pt"
    save_class_statistics(statistics_path, statistics_signature, counts)
    cached_counts = load_class_statistics(statistics_path, statistics_signature)
    assert cached_counts is not None
    assert all(torch.equal(counts[name], cached_counts[name]) for name in counts)
    assert load_class_statistics(statistics_path, "different") is None

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_bc_checkpoint(
        checkpoint_path,
        model,
        None,
        0,
        {"validation": {"loss": 1.0}},
        {"train": []},
        class_counts=counts,
        class_statistics_signature=statistics_signature,
    )
    restored, checkpoint = load_bc_checkpoint(str(checkpoint_path))
    assert checkpoint["epoch"] == 0
    assert restored.config.feature_channels == 16
    restored_counts = checkpoint_class_statistics(checkpoint, statistics_signature)
    assert restored_counts is not None
    assert all(torch.equal(counts[name], restored_counts[name]) for name in counts)


def test_replay_disk_cache_avoids_reparsing_json(tmp_path, monkeypatch):
    replay = {
        "rewards": [1, 0],
        "steps": [
            [
                {"observation": {"width": 16, "height": 16, "updates": _updates()}},
                {"observation": {}},
            ],
            [{"action": ["m u_1 n"], "step": 1}, {"action": [], "step": 1}],
        ],
    }
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    cache_dir = tmp_path / "replay_cache"

    result = prepare_replay_cache([replay_path, replay_path], cache_dir)
    assert result == {"replay_count": 1, "created_count": 1}
    assert len(list(cache_dir.glob("*.pickle"))) == 1

    def fail_json_load(_file: object) -> None:
        raise AssertionError("replay JSON was parsed after the cache was prepared")

    monkeypatch.setattr("luxai2021.imitation.data.json.load", fail_json_load)
    dataset = LuxReplayDataset([replay_path], replay_cache_dir=cache_dir)
    assert len(dataset) == 1
    assert dataset[0]["observation"].shape == (55, 32, 32)


def test_encoder_masks_padding_and_returns_global_features():
    snapshot = snapshot_from_updates(_updates(), width=16, height=16, turn=30)
    observation = torch.from_numpy(encode_snapshot(snapshot, 0))[None]
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=8, feature_channels=16))
    output = model(observation, return_features=True)

    features = output["features"]
    assert features.shape == (1, 16, 32, 32)
    assert not features[:, :, :8].any()
    assert not features[:, :, 24:].any()
    assert output["global_features"].shape == (1, model.encoder.global_output_channels)


@pytest.mark.parametrize(
    "encoder_type",
    ["unet", "resnet17x32", "resnet17x48", "resattn8", "transformer16", "axial32", "axial32_4m5"],
)
@pytest.mark.parametrize("board_size", BOARD_SIZES)
def test_all_encoders_mask_padding_and_return_finite_features(encoder_type, board_size):
    offset = (32 - board_size) // 2
    observation = torch.zeros(1, len(FEATURE_NAMES), 32, 32)
    observation[:, FEATURE_INDEX["board_mask"], offset : offset + board_size, offset : offset + board_size] = 1
    observation[:, FEATURE_INDEX["own_worker"], offset, offset] = 1
    model = LuxBehaviorCloningModel(
        ModelConfig(
            base_channels=8,
            feature_channels=16,
            encoder_type=encoder_type,
            transformer_dim=16,
            transformer_heads=4,
            transformer_ffn_dim=32,
            transformer16_layers=1,
            axial32_layers=1,
            axial32_4m5_dim=16,
            axial32_4m5_ffn_dim=32,
            axial32_4m5_layers=1,
            resattn8_base_channels=4,
            resattn8_feature_channels=16,
            resattn8_heads=4,
            resattn8_ffn_dim=32,
            resattn8_layers=1,
        )
    )

    output = model(observation, return_features=True)
    features = output["features"]
    mask = observation[:, FEATURE_INDEX["board_mask"] : FEATURE_INDEX["board_mask"] + 1]

    assert features.shape == (1, model.encoder.output_channels, 32, 32)
    assert torch.isfinite(features).all()
    assert torch.isfinite(output["global_features"]).all()
    assert not features.masked_select(mask == 0).any()
    output["worker_type"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_axis_attention_handles_fully_padded_groups_and_ignores_invalid_keys():
    attention = AxisAttention(channels=16, heads=4, dropout=0.0)
    distance_bias = torch.zeros(4, 32)
    inputs = torch.randn(1, 2, 4, 16)
    valid_mask = torch.tensor([[[True, True, False, False], [False, False, False, False]]])

    output = attention(inputs, valid_mask, distance_bias)
    changed = inputs.clone()
    changed[:, 0, 2:] = 1e6
    changed_output = attention(changed, valid_mask, distance_bias)

    assert torch.isfinite(output).all()
    assert not output[:, 1].any()
    assert torch.allclose(output[:, 0, :2], changed_output[:, 0, :2])


def test_global_spatial_attention_masks_padding_and_empty_samples():
    block = GlobalSpatialAttentionBlock(channels=16, heads=4, ffn_dim=32, dropout=0.0).eval()
    inputs = torch.randn(2, 16, 4, 4)
    mask = torch.zeros(2, 1, 4, 4)
    mask[0, :, 1:3, 1:3] = 1

    output = block(inputs, mask)
    changed = inputs.clone()
    changed[0] = torch.where(mask[0].bool(), changed[0], torch.full_like(changed[0], 1e6))
    changed_output = block(changed, mask)

    assert torch.isfinite(output).all()
    assert not output[1].any()
    assert not output.masked_select(mask == 0).any()
    assert torch.allclose(output[0], changed_output[0])


def test_axis_attention_sdpa_is_finite_for_large_fp16_inputs():
    attention = AxisAttention(channels=4, heads=1, dropout=0.0).half()
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.bias.zero_()
        identity = torch.eye(4, dtype=torch.float16)
        for offset in (0, 4, 8):
            attention.qkv.weight[offset : offset + 4].copy_(identity)
        attention.projection.weight.copy_(identity)
        attention.projection.bias.zero_()

    inputs = torch.full((1, 1, 2, 4), 300.0, dtype=torch.float16)
    output = attention(
        inputs,
        torch.ones(1, 1, 2, dtype=torch.bool),
        torch.zeros(1, 32),
    )

    assert torch.isfinite(output).all()


def test_axial_block_shares_one_distance_bias_between_axes():
    block = AxialTransformerBlock(channels=16, heads=4, ffn_dim=32, dropout=0.0)
    distance_bias_names = [name for name, _ in block.named_parameters() if "distance_bias" in name]
    assert distance_bias_names == ["distance_bias"]


def test_default_transformer_parameter_counts_match_unet_budget():
    parameter_counts = {
        encoder_type: sum(
            parameter.numel()
            for parameter in LuxBehaviorCloningModel(ModelConfig(encoder_type=encoder_type)).parameters()
        )
        for encoder_type in ("unet", "transformer16", "axial32")
    }
    baseline = parameter_counts["unet"]
    assert baseline == 8_383_254
    assert all(0.9 * baseline <= count <= 1.1 * baseline for count in parameter_counts.values())


def test_compact_axial_has_4m5_parameters_for_flat_distillation_policy():
    config = ModelConfig(
        encoder_type="axial32_4m5",
        policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT,
    )
    parameter_count = sum(parameter.numel() for parameter in LuxBehaviorCloningModel(config).parameters())

    assert parameter_count == 4_505_692


def test_resattn8_has_expected_flat_policy_parameter_count():
    config = ModelConfig(
        encoder_type="resattn8",
        policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT,
    )
    parameter_count = sum(parameter.numel() for parameter in LuxBehaviorCloningModel(config).parameters())

    assert parameter_count == 4_409_500


@pytest.mark.parametrize(
    ("encoder_type", "expected_parameter_count"),
    [("resnet17x32", 364_700), ("resnet17x48", 781_116)],
)
def test_resnet17_has_expected_flat_policy_parameter_count(encoder_type, expected_parameter_count):
    config = ModelConfig(
        encoder_type=encoder_type,
        policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT,
    )
    parameter_count = sum(parameter.numel() for parameter in LuxBehaviorCloningModel(config).parameters())

    assert parameter_count == expected_parameter_count


@pytest.mark.parametrize(
    "encoder_type",
    ["unet", "resnet17x32", "resnet17x48", "resattn8", "transformer16", "axial32", "axial32_4m5"],
)
def test_encoder_checkpoint_round_trip(tmp_path, encoder_type):
    config = ModelConfig(
        base_channels=8,
        feature_channels=16,
        encoder_type=encoder_type,
        transformer_dim=16,
        transformer_heads=4,
        transformer_ffn_dim=32,
        transformer16_layers=1,
        axial32_layers=1,
        axial32_4m5_dim=16,
        axial32_4m5_ffn_dim=32,
        axial32_4m5_layers=1,
        resattn8_base_channels=4,
        resattn8_feature_channels=16,
        resattn8_heads=4,
        resattn8_ffn_dim=32,
        resattn8_layers=1,
    )
    model = LuxBehaviorCloningModel(config)
    checkpoint_path = tmp_path / f"{encoder_type}.pt"
    save_bc_checkpoint(checkpoint_path, model, None, 0, {}, {"train": []})

    restored, _ = load_bc_checkpoint(str(checkpoint_path))

    assert restored.config == config


def test_checkpoint_saves_scaler_state_and_old_checkpoint_may_omit_it(tmp_path):
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=8, feature_channels=16))
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128.0)
    checkpoint_path = tmp_path / "with_scaler.pt"
    save_bc_checkpoint(
        checkpoint_path,
        model,
        None,
        0,
        {},
        {"train": []},
        scaler=scaler,
    )

    _, checkpoint = load_bc_checkpoint(str(checkpoint_path))
    assert checkpoint["scaler"] == scaler.state_dict()

    checkpoint.pop("scaler")
    torch.save(checkpoint, checkpoint_path)
    _, restored_old_checkpoint = load_bc_checkpoint(str(checkpoint_path))
    assert "scaler" not in restored_old_checkpoint


def test_checkpoint_rejects_nonfinite_model_parameters(tmp_path):
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=8, feature_channels=16))
    checkpoint_path = tmp_path / "nonfinite.pt"
    save_bc_checkpoint(checkpoint_path, model, None, 0, {}, {"train": []})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    first_parameter = next(iter(checkpoint["model"].values()))
    first_parameter.view(-1)[0] = float("nan")
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="non-finite model parameters"):
        load_bc_checkpoint(str(checkpoint_path))


def test_old_unet_checkpoint_without_encoder_config_loads(tmp_path):
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=8, feature_channels=16))
    checkpoint_path = tmp_path / "old_unet.pt"
    save_bc_checkpoint(checkpoint_path, model, None, 0, {}, {"train": []})
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for name in (
        "encoder_type",
        "transformer_dim",
        "transformer_heads",
        "transformer_ffn_dim",
        "transformer_dropout",
        "transformer16_layers",
        "axial32_layers",
        "axial32_4m5_dim",
        "axial32_4m5_ffn_dim",
        "axial32_4m5_layers",
        "resattn8_base_channels",
        "resattn8_feature_channels",
        "resattn8_heads",
        "resattn8_ffn_dim",
        "resattn8_layers",
    ):
        checkpoint["model_config"].pop(name)
    torch.save(checkpoint, checkpoint_path)

    restored, _ = load_bc_checkpoint(str(checkpoint_path))

    assert restored.config.encoder_type == "unet"


def test_resume_rejects_explicit_encoder_mismatch(tmp_path, monkeypatch):
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=8, feature_channels=16))
    checkpoint_path = tmp_path / "unet.pt"
    save_bc_checkpoint(checkpoint_path, model, None, 0, {}, {"train": []})
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_bc.py",
            "--replay-dir",
            "unused",
            "--output-dir",
            str(tmp_path / "output"),
            "--resume",
            str(checkpoint_path),
            "--encoder-type",
            "axial32",
            "--device",
            "cpu",
        ],
    )

    with pytest.raises(ValueError, match="Checkpoint encoder is unet"):
        train_main()


def test_compile_defaults_to_cuda_and_accepts_override():
    assert resolve_compile(enabled=None, device=torch.device("cuda"))
    assert not resolve_compile(enabled=None, device=torch.device("cpu"))
    assert not resolve_compile(enabled=False, device=torch.device("cuda"))


def test_amp_dtype_defaults_are_resolved_and_bfloat16_requires_device_support(monkeypatch):
    assert resolve_amp_dtype("bfloat16", torch.device("cpu")) == torch.bfloat16
    assert resolve_amp_dtype("float16", torch.device("cuda")) == torch.float16
    assert resolve_amp_dtype("float32", torch.device("cuda")) == torch.float32
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(ValueError, match="does not support bfloat16"):
        resolve_amp_dtype("bfloat16", torch.device("cuda"))


def test_parse_evaluation_checkpoint():
    name, path = parse_checkpoint("axial=models/axial/best.pt")
    assert name == "axial"
    assert str(path) == "models/axial/best.pt"
    with pytest.raises(ValueError, match="Expected NAME=CHECKPOINT"):
        parse_checkpoint("missing-name-separator")


def test_match_workers_and_seed_shards_are_capped_complete_and_balanced(monkeypatch):
    monkeypatch.setattr("examples.evaluate_bc_checkpoints.os.cpu_count", lambda: 16)
    assert parse_match_workers("auto") == "auto"
    assert parse_match_workers("3") == 3
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        parse_match_workers("0")

    assert resolve_match_workers("auto", torch.device("cuda"), 50) == 2
    assert resolve_match_workers("auto", torch.device("cpu"), 50) == 4
    assert resolve_match_workers(8, torch.device("cpu"), 3) == 3
    shards = shard_match_seeds(10, 11, 4)
    assert sorted(seed for shard in shards for seed in shard) == list(range(10, 21))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_round_robin_schedule_count_includes_pairs_seeds_and_both_orientations():
    assignments = match_pair_orientations(3)
    assert assignments == [
        (0, 0, 1, 0),
        (0, 0, 1, 1),
        (1, 0, 2, 0),
        (1, 0, 2, 1),
        (2, 1, 2, 0),
        (2, 1, 2, 1),
    ]
    assert expected_match_game_count(3, 50) == 300
    assert expected_match_game_count(3, 2) == 12
    assert select_evaluation_winner("lowest-loss", None) == ("test_loss", "lowest-loss")
    match_evaluation = {"standings": [{"name": "match-winner"}]}
    assert select_evaluation_winner("lowest-loss", match_evaluation) == (
        "round_robin_score_rate",
        "match-winner",
    )


def _synthetic_match(pair, winner, *, seed=0, orientation=0):
    return {
        "pair_index": 0,
        "pair": list(pair),
        "seed": seed,
        "orientation": orientation,
        "winner": winner,
        "draw": winner is None,
    }


def test_match_aggregation_uses_head_to_head_then_test_loss_for_ties():
    games = [
        _synthetic_match(("a", "b"), "a"),
        _synthetic_match(("a", "c"), "c"),
        _synthetic_match(("b", "c"), "b"),
        _synthetic_match(("a", "d"), "d"),
        _synthetic_match(("b", "d"), "d"),
        _synthetic_match(("c", "d"), "c"),
    ]
    pairwise, standings = aggregate_match_results(games, {"a": 0.4, "b": 0.1, "c": 0.3, "d": 0.2})
    by_name = {row["name"]: row for row in standings}
    assert by_name["a"]["wins"] == 1
    assert by_name["a"]["losses"] == 2
    assert by_name["a"]["draws"] == 0
    assert by_name["a"]["score_rate"] == pytest.approx(1 / 3)
    assert [row["name"] for row in standings] == ["c", "d", "a", "b"]
    assert len(pairwise) == 6
    reversed_pairwise, reversed_standings = aggregate_match_results(
        list(reversed(games)),
        {"a": 0.4, "b": 0.1, "c": 0.3, "d": 0.2},
    )
    assert reversed_pairwise == pairwise
    assert reversed_standings == standings

    draw_games = [_synthetic_match(("a", "b"), None)]
    _, draw_standings = aggregate_match_results(draw_games, {"a": 0.2, "b": 0.1})
    assert [row["name"] for row in draw_standings] == ["b", "a"]
    assert draw_standings[0]["draws"] == 1
    assert draw_standings[0]["score_rate"] == 0.5
    _, name_tiebreak_standings = aggregate_match_results(draw_games, {"a": 0.1, "b": 0.1})
    assert [row["name"] for row in name_tiebreak_standings] == ["a", "b"]


def test_match_result_sorting_is_independent_of_worker_completion_order():
    games = [
        {"pair_index": 1, "seed": 4, "orientation": 1},
        {"pair_index": 0, "seed": 5, "orientation": 0},
        {"pair_index": 0, "seed": 4, "orientation": 1},
        {"pair_index": 0, "seed": 4, "orientation": 0},
    ]
    expected = [(0, 4, 0), (0, 4, 1), (0, 5, 0), (1, 4, 1)]
    assert [(game["pair_index"], game["seed"], game["orientation"]) for game in sort_match_games(games)] == expected
    assert sort_match_games(list(reversed(games))) == sort_match_games(games)


def test_gradient_accumulation_steps_partial_final_group():
    dataset = LuxReplayDataset(
        ["luxai2021/tests/replays_for_test/27095556.json"],
        max_turns=5,
    )
    loader = DataLoader(dataset, batch_size=1)
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=4, feature_channels=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    metrics = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        {},
        optimizer=optimizer,
        scaler=scaler,
        show_progress=False,
        gradient_accumulation_steps=2,
    )

    assert metrics["optimizer_steps"] == 3
    assert metrics["samples"] == 5


def test_run_epoch_rejects_nonfinite_loss_before_optimizer_update():
    dataset = LuxReplayDataset(
        ["luxai2021/tests/replays_for_test/27095556.json"],
        max_turns=1,
    )
    loader = DataLoader(dataset, batch_size=1)
    model = LuxBehaviorCloningModel(ModelConfig(base_channels=4, feature_channels=8))
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        run_epoch(
            model,
            loader,
            torch.device("cpu"),
            {},
            optimizer=optimizer,
            scaler=scaler,
            show_progress=False,
        )


def _comparison_metrics(validation_loss, split_signature="same"):
    phase = {
        "loss": validation_loss,
        "worker_active_accuracy": 0.5,
        "cart_active_accuracy": 0.4,
        "city_active_accuracy": 0.3,
        "samples_per_second": 10.0,
        "peak_cuda_memory_allocated_bytes": 1024,
    }
    return {
        "model_config": {"encoder_type": "unet"},
        "model_parameter_count": 100,
        "encoder_parameter_count": 80,
        "data_split_signature": split_signature,
        "class_statistics_signature": "classes",
        "training_config": {"seed": 42},
        "history": [{"epoch": 0, "train": phase, "validation": phase}],
        "test": {"loss": validation_loss},
    }


def test_architecture_comparison_ranks_loss_and_rejects_mismatched_split():
    comparison = build_comparison(
        {
            "higher": _comparison_metrics(2.0),
            "lower": _comparison_metrics(1.0),
        }
    )
    assert comparison["winner"] == "lower"
    with pytest.raises(ValueError, match="data_split_signature"):
        build_comparison(
            {
                "first": _comparison_metrics(1.0),
                "second": _comparison_metrics(1.0, split_signature="different"),
            }
        )


def test_architecture_comparison_allows_encoder_and_output_path_to_differ():
    first = _comparison_metrics(1.0)
    second = _comparison_metrics(1.1)
    first["training_config"] = {"encoder_type": "resnet17x32", "output_dir": "first", "seed": 42}
    second["training_config"] = {"encoder_type": "resnet17x48", "output_dir": "second", "seed": 42}

    comparison = build_comparison({"first": first, "second": second})

    assert comparison["training_config"] == {"seed": 42}


def test_replay_uses_next_step_actions():
    dataset = LuxReplayDataset(
        ["luxai2021/tests/replays_for_test/27095556.json"],
        max_turns=1,
    )
    sample = dataset[0]
    action_targets = sample["worker_type"]
    assert (action_targets == 1).sum() == 1


def test_game_encoder_and_agent_smoke(tmp_path):
    configs = LuxMatchConfigs_Default.copy()
    configs["seed"] = 123
    game = Game(configs)
    snapshot = snapshot_from_game(game)
    features = encode_snapshot(snapshot, 0)
    assert features[FEATURE_INDEX["own_worker"]].sum() == 1

    model = LuxBehaviorCloningModel(ModelConfig(base_channels=8, feature_channels=16))
    checkpoint_path = tmp_path / "agent.pt"
    save_bc_checkpoint(checkpoint_path, model, None, 0, {}, {"train": []})
    agent = BehaviorCloningAgent(str(checkpoint_path), device="cpu")
    actions = agent.process_turn(game, 0)

    assert isinstance(actions, list)
    assert all(action.team == 0 for action in actions)
    validated = []
    accumulated_stats = {0: {}, 1: {}}
    for action in actions:
        assert action.is_valid(game, validated, accumulated_stats)
        validated.append(action)
        accumulated_stats = action.commit_action_update_stats(game, accumulated_stats)
