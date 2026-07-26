# ruff: noqa: ANN001, ANN201, ANN202, PLR2004, S101, SLF001

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from luxai2021.game.actions import SpawnWorkerAction
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
    discover_sources,
)
from luxai2021.imitation.masking import (
    apply_legal_action_mask,
    build_legal_masks,
)
from luxai2021.imitation.model import (
    LuxBehaviorCloningModel,
    ModelConfig,
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


def test_source_dataset_selects_source_submission_even_when_it_loses(tmp_path):
    source_id = 12345
    source_root = tmp_path / str(source_id)
    replay_root = source_root / "game"
    replay_root.mkdir(parents=True)
    (source_root / "agent_info.json").write_text(
        json.dumps({"agent_id": source_id, "lb": 1900.0}),
        encoding="utf-8",
    )
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
    replay_path = replay_root / "game.json"
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    (replay_root / "game_info.json").write_text(
        json.dumps(
            {
                "agents": [
                    {"index": 0, "submissionId": 999, "reward": 1},
                    {"index": 1, "submissionId": source_id, "reward": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    sources = discover_sources([replay_path])
    dataset = LuxReplayDataset(
        [replay_path],
        team_selection="source",
        source_ids=(source_id,),
        max_turns=1,
    )

    assert [(source.source_id, source.lb) for source in sources] == [(source_id, 1900.0)]
    assert dataset.samples[0][2:] == (1, 0)
    assert dataset[0]["source_index"].item() == 0
    assert (dataset[0]["worker_type"] == WORKER_ACTIONS.index("move")).sum() == 1


def test_source_dataset_uses_both_players_for_source_self_play(tmp_path):
    source_id = 12345
    source_root = tmp_path / str(source_id)
    replay_root = source_root / "game"
    replay_root.mkdir(parents=True)
    (source_root / "agent_info.json").write_text(
        json.dumps({"agent_id": source_id, "lb": 1900.0}),
        encoding="utf-8",
    )
    replay_path = replay_root / "game.json"
    replay_path.write_text(json.dumps({"steps": [], "step": 1}), encoding="utf-8")
    (replay_root / "game_info.json").write_text(
        json.dumps(
            {
                "agents": [
                    {"index": 0, "submissionId": source_id, "reward": 1},
                    {"index": 1, "submissionId": source_id, "reward": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = LuxReplayDataset([replay_path], team_selection="source", source_ids=(source_id,))
    assert [sample[2] for sample in dataset.samples] == [0, 1]


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


def test_durrett_encoder_masks_padding_and_routes_source_heads():
    snapshot = snapshot_from_updates(_updates(), width=16, height=16, turn=30)
    observation = torch.from_numpy(encode_snapshot(snapshot, 0))[None]
    config = ModelConfig(
        feature_channels=16,
        encoder_type="durrett",
        durrett_layers=1,
        transformer_layers=1,
        transformer_heads=1,
        source_ids=(11, 22),
    )
    model = LuxBehaviorCloningModel(config)
    output = model(observation, source_index=torch.tensor([0]), return_features=True)

    assert output["worker_type"].shape == (1, len(WORKER_ACTIONS), 32, 32)
    assert output["features"].shape == (1, 16, 32, 32)
    assert not output["features"][:, :, :8].any()
    assert not output["features"][:, :, 24:].any()
    output["worker_type"].sum().backward()
    classifiers = model.heads["worker_type"].classifiers
    assert classifiers[0].weight.grad is not None
    assert classifiers[1].weight.grad is None

    model.zero_grad(set_to_none=True)
    mixed_output = model(observation.repeat(2, 1, 1, 1), source_index=torch.tensor([0, 1]))
    mixed_output["worker_type"].sum().backward()
    assert classifiers[0].weight.grad is not None
    assert classifiers[1].weight.grad is not None


def test_source_checkpoint_agent_default_and_override(tmp_path):
    model = LuxBehaviorCloningModel(
        ModelConfig(
            feature_channels=16,
            encoder_type="durrett",
            durrett_layers=1,
            source_ids=(11, 22),
        )
    )
    checkpoint_path = tmp_path / "durrett.pt"
    save_bc_checkpoint(
        checkpoint_path,
        model,
        None,
        0,
        {},
        {"train": []},
        training_profile="durrett",
        source_catalog=(
            {"source_id": 11, "lb": 1800.0, "index": 0},
            {"source_id": 22, "lb": 1900.0, "index": 1},
        ),
        default_source_id=22,
    )

    default_agent = BehaviorCloningAgent(str(checkpoint_path), device="cpu")
    override_agent = BehaviorCloningAgent(str(checkpoint_path), device="cpu", source_id=11)
    assert (default_agent.source_id, default_agent.source_index) == (22, 1)
    assert (override_agent.source_id, override_agent.source_index) == (11, 0)
    with pytest.raises(ValueError, match="available source IDs"):
        BehaviorCloningAgent(str(checkpoint_path), device="cpu", source_id=99)


def test_durrett_city_interpreter_prioritizes_worker_builds():
    tiles = []
    city_cells = []
    for x in (1, 2):
        tile = SimpleNamespace(
            pos=SimpleNamespace(x=x, y=1),
            can_act=lambda: True,
            get_tile_id=lambda x=x: f"c_{x}_1",
        )
        tiles.append(tile)
        city_cells.append(SimpleNamespace(city_tile=tile))
    game = SimpleNamespace(
        cities={"c": SimpleNamespace(team=0, city_cells=city_cells)},
        state={"teamStates": {0: {"units": {}}, 1: {"units": {}}}},
    )
    snapshot = snapshot_from_updates(
        [
            "rp 0 0",
            "c 0 c 100 23",
            "ct 0 c 1 1 0",
            "ct 0 c 2 1 0",
            "D_DONE",
        ],
        width=16,
        height=16,
        turn=0,
    )
    output = {"city": torch.zeros(1, len(CITY_ACTIONS), 32, 32)}
    output["city"][:, CITY_ACTIONS.index("no_action")] = 10
    output["city"][:, CITY_ACTIONS.index("build_worker")] = -10
    agent = object.__new__(BehaviorCloningAgent)
    agent.training_profile = "durrett"

    actions = agent._city_actions(game, 0, snapshot, output, *snapshot.padding)

    assert len(actions) == 2
    assert all(isinstance(action, SpawnWorkerAction) for action in actions)


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
