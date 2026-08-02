# ruff: noqa: PLR2004, S101
from __future__ import annotations

from luxai2021.game.actions import MoveAction, ResearchAction
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from luxai2021.imitation.schema import snapshot_from_game, snapshot_from_updates
from luxai2021.imitation.selfplay import KaggleReplayRecorder, kaggle_updates_from_game


def create_game(seed: int = 42) -> Game:
    config = dict(LuxMatchConfigs_Default)
    config["seed"] = seed
    return Game(config)


def test_kaggle_updates_round_trip_live_game_state() -> None:
    game = create_game()
    expected = snapshot_from_game(game)
    actual = snapshot_from_updates(kaggle_updates_from_game(game), game.map.width, game.map.height, 0)

    assert actual == expected


def test_recorder_places_turn_actions_on_following_kaggle_step() -> None:
    game = create_game(seed=7)
    recorder = KaggleReplayRecorder()
    recorder.record_turn(
        game,
        [
            MoveAction(team=0, unit_id="u_1", direction="n"),
            ResearchAction(team=1, x=2, y=3, unit_id=None),
        ],
    )
    game.last_winning_team = 1

    replay = recorder.build_replay(
        game,
        seed=7,
        teacher_sha256="teacher-sha",
        tta="rot180",
        team_names=("teacher.pt", "teacher.pt"),
    )

    assert len(replay["steps"]) == 2
    assert replay["steps"][0][0]["action"] == []
    assert replay["steps"][1][0]["action"] == ["m u_1 n"]
    assert replay["steps"][1][1]["action"] == ["r 2 3"]
    assert replay["rewards"] == [0.0, 1.0]
    assert replay["info"]["tta"] == "rot180"
    assert replay["configuration"]["seed"] == 7
