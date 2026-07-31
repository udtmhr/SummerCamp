# ruff: noqa: ANN001, ANN201, ANN202, ANN204, PLR2004, S101, SLF001

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.train_distilled_bc import make_loader
from luxai2021.imitation.actions import (
    FIRST_PLACE_ACTION_SCHEMA,
    FIRST_PLACE_CART_ACTIONS,
    FIRST_PLACE_WORKER_ACTIONS,
    first_place_action_remap,
)
from luxai2021.imitation.agent import BehaviorCloningAgent
from luxai2021.imitation.data import IGNORE_INDEX, MAX_ENTITIES, _winner_from_rewards
from luxai2021.imitation.distillation import (
    DISTILLATION_CACHE_VERSION,
    DISTILLATION_PREPARED_CACHE_VERSION,
    LuxDistillationDataset,
    augment_distillation_batch,
    cache_matches_replay,
    compact_distillation_collate,
    distillation_loss,
    prepared_distillation_cache_path,
    replay_fingerprint,
    save_prepared_distillation_cache,
)
from luxai2021.imitation.first_place import (
    FIRST_PLACE_TEACHER_SHA256,
    augment_first_place_sample,
    build_first_place_targets,
)
from luxai2021.imitation.model import (
    POLICY_SCHEMA_FIRST_PLACE_FLAT,
    LuxBehaviorCloningModel,
    ModelConfig,
    load_bc_checkpoint,
    save_bc_checkpoint,
)
from luxai2021.imitation.schema import FEATURE_NAMES, encode_snapshot, snapshot_from_updates


def _snapshot():
    updates = [
        "rp 0 50",
        "rp 1 0",
        "u 0 0 worker 1 1 0 10 0 0",
        "u 0 0 receiver 2 1 0 0 0 0",
        "u 1 0 cart 3 3 0 0 0 0",
        "c 0 city 100 23",
        "ct 0 city 1 2 0",
        "ccd 1 1 2",
        "D_DONE",
    ]
    return snapshot_from_updates(updates, width=16, height=16, turn=0)


def _small_flat_config(encoder_type):
    return ModelConfig(
        base_channels=8,
        feature_channels=16,
        encoder_type=encoder_type,
        transformer_dim=16,
        transformer_heads=4,
        transformer_ffn_dim=32,
        transformer16_layers=1,
        axial32_layers=1,
        policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT,
    )


def test_first_place_flat_action_order_and_rotation():
    assert len(FIRST_PLACE_WORKER_ACTIONS) == 19
    assert len(FIRST_PLACE_CART_ACTIONS) == 17
    assert FIRST_PLACE_WORKER_ACTIONS[1:5] == ("move_n", "move_e", "move_s", "move_w")
    assert FIRST_PLACE_WORKER_ACTIONS[5:9] == (
        "transfer_wood_n",
        "transfer_wood_e",
        "transfer_wood_s",
        "transfer_wood_w",
    )
    assert FIRST_PLACE_WORKER_ACTIONS[-2:] == ("pillage", "build_city")
    remap = first_place_action_remap("worker", 2)
    assert remap[FIRST_PLACE_WORKER_ACTIONS.index("move_n")] == FIRST_PLACE_WORKER_ACTIONS.index("move_s")
    assert remap[FIRST_PLACE_WORKER_ACTIONS.index("transfer_coal_e")] == FIRST_PLACE_WORKER_ACTIONS.index(
        "transfer_coal_w"
    )


def test_teacher_cache_validates_precision_configuration(tmp_path):
    replay_path = tmp_path / "replay.json"
    replay_path.write_text("{}", encoding="utf-8")
    cache = {
        "cache_version": DISTILLATION_CACHE_VERSION,
        "source": replay_fingerprint(replay_path),
        "teacher_sha256": FIRST_PLACE_TEACHER_SHA256,
        "rot180": True,
        "dtype": "float16",
        "amp_dtype": "bfloat16",
    }

    assert cache_matches_replay(
        cache,
        replay_path,
        FIRST_PLACE_TEACHER_SHA256,
        rot180=True,
        cache_dtype="float16",
        amp_dtype="bfloat16",
    )
    assert not cache_matches_replay(
        cache,
        replay_path,
        FIRST_PLACE_TEACHER_SHA256,
        rot180=True,
        cache_dtype="float32",
        amp_dtype="float32",
    )


@pytest.mark.parametrize("rewards", [[None, 0], [0, None], [None, None]])
def test_unknown_replay_rewards_have_no_winner(rewards):
    assert _winner_from_rewards(rewards) is None


def test_distillation_loader_does_not_pad_tail_batch():
    samples = []
    for entity_count in range(1, 6):
        sample = {"observation": torch.zeros(1), "sample_weight": torch.tensor(1.0)}
        for entity, actions in FIRST_PLACE_ACTION_SCHEMA.items():
            sample[f"{entity}_positions"] = torch.zeros(entity_count, 2, dtype=torch.long)
            sample[f"{entity}_flat"] = torch.zeros(entity_count, dtype=torch.long)
            sample[f"{entity}_legal_mask"] = torch.ones(entity_count, len(actions), dtype=torch.bool)
            sample[f"{entity}_teacher_logits"] = torch.zeros(entity_count, len(actions))
        samples.append(sample)
    loader = make_loader(
        samples,
        4,
        training=False,
        num_workers=0,
        prefetch_factor=1,
        seed=42,
    )

    assert [batch["observation"].shape[0] for batch in loader] == [4, 1]


def test_prepared_distillation_cache_loads_compact_samples(tmp_path):
    replay_path = tmp_path / "replay.json"
    replay_path.write_text("{}", encoding="utf-8")
    prepared_dir = tmp_path / "prepared"
    arrays = {
        "cache_version": np.asarray(DISTILLATION_PREPARED_CACHE_VERSION),
        "turn_count": np.asarray(1),
        "winner": np.asarray(0),
        "observation_dtype": np.asarray("float16"),
        "observation": np.zeros((1, 2, len(FEATURE_NAMES), 32, 32), dtype=np.float16),
    }
    for entity, actions in FIRST_PLACE_ACTION_SCHEMA.items():
        arrays[f"{entity}_offsets"] = np.asarray([0, 1, 2])
        arrays[f"{entity}_positions"] = np.asarray([[1, 2], [3, 4]], dtype=np.int16)
        arrays[f"{entity}_flat"] = np.asarray([0, 0], dtype=np.int8)
        arrays[f"{entity}_legal_mask"] = np.ones((2, len(actions)), dtype=np.bool_)
        arrays[f"{entity}_teacher_logits"] = np.zeros((2, len(actions)), dtype=np.float16)
    path = prepared_distillation_cache_path(replay_path, prepared_dir, "float16")
    save_prepared_distillation_cache(path, arrays)

    dataset = LuxDistillationDataset(
        [replay_path],
        tmp_path / "unused-teacher",
        prepared_cache_dir=prepared_dir,
    )

    assert len(dataset) == 2
    assert dataset[0]["observation"].dtype == torch.float16
    assert dataset[0]["worker_positions"].tolist() == [[1, 2]]
    assert dataset[0]["sample_weight"].item() == 1.5
    assert dataset[1]["sample_weight"].item() == 1.0


@pytest.mark.parametrize("transform_id", range(8))
def test_batched_augmentation_matches_numpy_d4(transform_id):
    snapshot = _snapshot()
    observation = encode_snapshot(snapshot, 0)
    targets = build_first_place_targets(snapshot, 0, ["m worker n"])
    sample = {"observation": torch.from_numpy(observation), "sample_weight": torch.tensor(1.0)}
    for entity, actions in FIRST_PLACE_ACTION_SCHEMA.items():
        count = int(np.count_nonzero(targets[f"{entity}_flat"] != IGNORE_INDEX))
        teacher = np.arange(MAX_ENTITIES * len(actions), dtype=np.float32).reshape(MAX_ENTITIES, -1)
        targets[f"{entity}_teacher_logits"] = teacher
        for suffix in ("positions", "flat", "legal_mask", "teacher_logits"):
            sample[f"{entity}_{suffix}"] = torch.from_numpy(targets[f"{entity}_{suffix}"][:count])

    batch = compact_distillation_collate([sample])
    actual = augment_distillation_batch(batch, torch.tensor([transform_id]))
    rotations, horizontal_flip = divmod(transform_id, 2)
    expected_observation, expected_targets = augment_first_place_sample(
        observation,
        targets,
        rotations,
        horizontal_flip=bool(horizontal_flip),
    )

    assert torch.equal(actual["observation"][0], torch.from_numpy(expected_observation))
    for entity in FIRST_PLACE_ACTION_SCHEMA:
        count = sample[f"{entity}_flat"].shape[0]
        for suffix in ("positions", "flat", "legal_mask", "teacher_logits"):
            expected = torch.from_numpy(expected_targets[f"{entity}_{suffix}"][:count])
            assert torch.equal(actual[f"{entity}_{suffix}"][0], expected)


def test_flat_targets_preserve_worker_pillage_and_transfer():
    snapshot = _snapshot()
    transfer = build_first_place_targets(snapshot, 0, ["t worker receiver wood 10"])
    pillage = build_first_place_targets(snapshot, 0, ["p worker"])

    assert transfer["worker_flat"][0] == FIRST_PLACE_WORKER_ACTIONS.index("transfer_wood_e")
    assert pillage["worker_flat"][0] == FIRST_PLACE_WORKER_ACTIONS.index("pillage")


def test_flat_augmentation_rotates_labels_masks_and_teacher_logits():
    snapshot = _snapshot()
    observation = encode_snapshot(snapshot, 0)
    targets = build_first_place_targets(snapshot, 0, ["m worker n"])
    teacher = np.arange(MAX_ENTITIES * len(FIRST_PLACE_WORKER_ACTIONS), dtype=np.float32).reshape(MAX_ENTITIES, -1)
    targets["worker_teacher_logits"] = teacher.copy()

    _, rotated = augment_first_place_sample(observation, targets, 1, horizontal_flip=False)

    north = FIRST_PLACE_WORKER_ACTIONS.index("move_n")
    west = FIRST_PLACE_WORKER_ACTIONS.index("move_w")
    assert rotated["worker_flat"][0] == west
    assert rotated["worker_teacher_logits"][0, west] == teacher[0, north]


@pytest.mark.parametrize("encoder_type", ["unet", "transformer16", "axial32"])
def test_all_student_encoders_support_flat_policy_checkpoint(tmp_path, encoder_type):
    model = LuxBehaviorCloningModel(_small_flat_config(encoder_type))
    observation = torch.zeros(1, len(FEATURE_NAMES), 32, 32)
    output = model(observation)

    assert {name: value.shape[1] for name, value in output.items()} == {
        name: len(actions) for name, actions in FIRST_PLACE_ACTION_SCHEMA.items()
    }
    checkpoint_path = tmp_path / f"{encoder_type}.pt"
    save_bc_checkpoint(
        checkpoint_path,
        model,
        None,
        0,
        {},
        {"train": []},
        extra_metadata={"inference_augmentation": "rot180"},
    )
    restored, checkpoint = load_bc_checkpoint(str(checkpoint_path))
    assert restored.config.policy_schema == POLICY_SCHEMA_FIRST_PLACE_FLAT
    assert checkpoint["inference_augmentation"] == "rot180"


def test_distillation_loss_is_finite_and_backpropagates():
    model = LuxBehaviorCloningModel(_small_flat_config("unet"))
    observation = torch.zeros(1, len(FEATURE_NAMES), 32, 32)
    batch = {"observation": observation, "sample_weight": torch.ones(1)}
    for entity, actions in FIRST_PLACE_ACTION_SCHEMA.items():
        positions = torch.full((1, MAX_ENTITIES, 2), -1, dtype=torch.long)
        positions[0, 0] = torch.tensor((8, 8))
        labels = torch.full((1, MAX_ENTITIES), IGNORE_INDEX, dtype=torch.long)
        labels[0, 0] = 0
        legal = torch.zeros((1, MAX_ENTITIES, len(actions)), dtype=torch.bool)
        legal[0, 0] = True
        teacher = torch.zeros((1, MAX_ENTITIES, len(actions)))
        teacher[0, 0, -1] = 2
        batch[f"{entity}_positions"] = positions
        batch[f"{entity}_flat"] = labels
        batch[f"{entity}_legal_mask"] = legal
        batch[f"{entity}_teacher_logits"] = teacher

    losses = distillation_loss(model(observation), batch)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_rot180_tta_restores_spatial_and_action_axes():
    agent = BehaviorCloningAgent.__new__(BehaviorCloningAgent)
    agent.model = SimpleNamespace(config=SimpleNamespace(policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT))
    source = torch.zeros(1, len(FIRST_PLACE_WORKER_ACTIONS), 32, 32)
    source[0, FIRST_PLACE_WORKER_ACTIONS.index("move_s"), 30, 29] = 7

    restored = agent._restore_rot180({"worker": source})["worker"]

    assert restored[0, FIRST_PLACE_WORKER_ACTIONS.index("move_n"), 1, 2] == 7


def test_rot180_tta_updates_coordinate_feature_semantics():
    class CaptureModel:
        config = SimpleNamespace(policy_schema=POLICY_SCHEMA_FIRST_PLACE_FLAT)

        def __call__(self, observation):
            self.observation = observation
            return {"worker": torch.zeros(observation.shape[0], len(FIRST_PLACE_WORKER_ACTIONS), 32, 32)}

    agent = BehaviorCloningAgent.__new__(BehaviorCloningAgent)
    agent.model = CaptureModel()
    agent.tta = "rot180"
    observation = torch.zeros(1, len(FEATURE_NAMES), 32, 32)
    x_index = FEATURE_NAMES.index("x_coordinate")
    y_index = FEATURE_NAMES.index("y_coordinate")
    observation[0, x_index, 1, 2] = 0.5
    observation[0, y_index, 1, 2] = -0.25

    agent._predict(observation)

    assert agent.model.observation[1, x_index, 30, 29] == -0.5
    assert agent.model.observation[1, y_index, 30, 29] == 0.25
