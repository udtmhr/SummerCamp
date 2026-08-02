from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from luxai2021.imitation.schema import snapshot_from_game

if TYPE_CHECKING:
    from collections.abc import Sequence

    from luxai2021.game.actions import Action


@dataclass(frozen=True)
class RecordedTurn:
    turn: int
    updates: tuple[str, ...]
    actions: tuple[tuple[str, ...], tuple[str, ...]]


def kaggle_updates_from_game(game: object) -> list[str]:
    """Serialize the live engine state into the updates consumed by Kaggle replays."""
    snapshot = snapshot_from_game(game)
    updates = [f"rp {team} {snapshot.research_points[team]}" for team in (0, 1)]
    updates.extend(
        f"r {resource_type} {x} {y} {amount}"
        for (x, y), (resource_type, amount) in sorted(
            snapshot.resources.items(), key=lambda item: (item[0][1], item[0][0])
        )
    )
    updates.extend(
        (
            f"u {unit.unit_type} {unit.team} {unit.unit_id} {unit.x} {unit.y} {unit.cooldown} "
            f"{unit.cargo['wood']} {unit.cargo['coal']} {unit.cargo['uranium']}"
        )
        for unit in sorted(snapshot.units.values(), key=lambda item: (item.team, item.unit_id))
    )
    for city_id, (team, fuel, upkeep) in sorted(snapshot.cities.items()):
        updates.append(f"c {team} {city_id} {fuel} {upkeep}")
    updates.extend(
        f"ct {tile.team} {tile.city_id} {tile.x} {tile.y} {tile.cooldown}"
        for tile in sorted(snapshot.city_tiles, key=lambda item: (item.team, item.city_id, item.y, item.x))
    )
    updates.extend(
        f"ccd {x} {y} {road}"
        for (x, y), road in sorted(snapshot.roads.items(), key=lambda item: (item[0][1], item[0][0]))
    )
    return updates


class KaggleReplayRecorder:
    """Record engine self-play in the Kaggle ``steps`` schema used by distillation."""

    def __init__(self) -> None:
        self.turns: list[RecordedTurn] = []

    def reset(self) -> None:
        self.turns.clear()

    def record_turn(self, game: object, actions: Sequence[Action]) -> None:
        turn = int(game.state["turn"])
        if turn != len(self.turns):
            message = f"Expected self-play turn {len(self.turns)}, got {turn}"
            raise ValueError(message)
        by_team: tuple[list[str], list[str]] = ([], [])
        for action in actions:
            if action.team not in (0, 1):
                message = f"Unexpected action team: {action.team}"
                raise ValueError(message)
            by_team[action.team].append(action.to_message(game))
        self.turns.append(
            RecordedTurn(
                turn=turn,
                updates=tuple(kaggle_updates_from_game(game)),
                actions=(tuple(by_team[0]), tuple(by_team[1])),
            )
        )

    def build_replay(
        self,
        game: object,
        *,
        seed: int,
        teacher_sha256: str,
        tta: str,
        team_names: tuple[str, str],
    ) -> dict[str, object]:
        if not self.turns:
            raise ValueError("Cannot build a replay without recorded turns")
        winner = game.last_winning_team
        if winner is None:
            winner = game.get_winning_team()
        if winner not in (0, 1):
            message = f"Unexpected winning team: {winner}"
            raise ValueError(message)
        rewards = [0.0, 0.0]
        rewards[winner] = 1.0
        final_updates = tuple(kaggle_updates_from_game(game))
        steps = []
        for step in range(len(self.turns) + 1):
            updates = self.turns[step].updates if step < len(self.turns) else final_updates
            actions = ((), ()) if step == 0 else self.turns[step - 1].actions
            status = "DONE" if step == len(self.turns) else "ACTIVE"
            entries = []
            for team in (0, 1):
                observation: dict[str, object] = {
                    "player": team,
                    "remainingOverageTime": 0,
                    "reward": rewards[team] if status == "DONE" else 0,
                }
                if team == 0:
                    observation.update(
                        {
                            "step": step,
                            "width": game.map.width,
                            "height": game.map.height,
                            "updates": list(updates),
                        }
                    )
                entries.append(
                    {
                        "action": list(actions[team]),
                        "info": {},
                        "observation": observation,
                        "reward": rewards[team] if status == "DONE" else 0,
                        "status": status,
                    }
                )
            steps.append(entries)
        return {
            "configuration": {
                "episodeSteps": len(self.turns) + 1,
                "height": game.map.height,
                "mapType": "random",
                "seed": seed,
                "width": game.map.width,
            },
            "description": "Lux AI 2021 first-place teacher self-play for policy distillation",
            "id": f"first-place-selfplay-seed-{seed}",
            "info": {
                "source": "first-place-selfplay",
                "teacher_sha256": teacher_sha256,
                "team_names": list(team_names),
                "tta": tta,
                "turn_count": len(self.turns),
            },
            "name": "lux_ai_2021",
            "rewards": rewards,
            "schema_version": 1,
            "statuses": ["DONE", "DONE"],
            "steps": steps,
            "title": f"First-place self-play seed {seed}",
            "version": "3.1.0",
        }
