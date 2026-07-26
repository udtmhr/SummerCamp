import json

from luxai2021.env.agent import Agent
from luxai2021.game.constants import LuxMatchConfigs_Default
from luxai2021.game.game import Game
from luxai2021.game.replay import Replay


def test_replay_records_agent_names_result_and_file(tmp_path):
    config = dict(LuxMatchConfigs_Default)
    config["seed"] = 42
    game = Game(config)

    first = Agent()
    first.set_team(0)
    first.replay_name = "best.pt"
    second = Agent()
    second.set_team(1)
    second.replay_name = "latest.pt"
    game.agents = [first, second]

    replay_path = tmp_path / "best-vs-latest_seed42.json"
    replay = Replay(game, str(replay_path))
    replay.write(game)

    data = json.loads(replay_path.read_text())
    assert [team["name"] for team in data["teamDetails"]] == ["best.pt", "latest.pt"]
    assert {rank["agentID"] for rank in data["results"]["ranks"]} == {0, 1}
    assert data["results"]["ranks"][0]["rank"] == 1
    assert data["results"]["ranks"][0]["agentID"] == game.last_winning_team
    assert data["results"]["replayFile"] == str(replay_path)
