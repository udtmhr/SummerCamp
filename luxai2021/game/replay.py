
import json
import os
from luxai2021.game.actions import Action
from typing import List
from .constants import Constants

class Replay:
    """
    Implements saving of replays. Loosely mirrors '/src/Replay/index.ts'
    """
    def __init__(self, game, file:str, stateful:bool=False):
        """
        Creates a replay-writer class to the target file. Optionally
        stateful which includes the whole map in each turn instead of
        just the actions taken.

        Args:
            file ([type]): [description]
            stateful (bool, optional): [description]. Defaults to False.
        """
        self.replayFilePath = None
        self.file = file
        self.stateful = stateful
        self.clear(game)
    
    
    def clear(self, game):
        team_details = [{"name": "Agent0", "tournamentID": ""}, {"name": "Agent1", "tournamentID": ""}]
        for agent in game.agents:
            if agent.team in (Constants.TEAM.A, Constants.TEAM.B):
                team_details[agent.team]["name"] = getattr(agent, "replay_name", type(agent).__name__)

        self.data = {
            'seed' : 0,
            'mapType' :  Constants.MAP_TYPES.RANDOM,
            'teamDetails' : team_details,
            'allCommands' : [], # Array<Array<str>>;
            'version' : "3.1.0", #string;
            "results": {"ranks": [], "replayFile": os.path.normpath(self.file)},
        }
        if "seed" in game.configs:
            self.data["seed"] = game.configs["seed"]
        if self.stateful:
            self.data['stateful'] = [] #Array<SerializedState>;

    def add_actions(self, game, actions: List[Action]) -> None:
        """
        Adds the specified commands to the replay.

        Args:
            commands (List[int]): Commands to add.
        """
        commands = []
        for action in actions:
            commands.append(
                {
                    "command" : action.to_message(self),
                    "agentID" : action.team,
                }
            )
                    
        self.data["allCommands"].append(commands)
        
    
    def add_state(self, game) -> None:
        """
        Write this state.

        Args:
            game (Game): [description]
        """
        if self.stateful:
            state = game.to_state_object()
            self.data['stateful'].append(state)
    
    def write(self, game) -> None:
        """
        Write this replay to the specified target file.
        """
        self.data['width'] = game.map.width
        self.data['height'] = game.map.height
        winner = game.last_winning_team
        if winner is None:
            winner = game.get_winning_team()
        game.last_winning_team = winner
        loser = Constants.TEAM.B if winner == Constants.TEAM.A else Constants.TEAM.A
        self.data["results"]["ranks"] = [
            {"rank": 1, "agentID": winner},
            {"rank": 2, "agentID": loser},
        ]

        with open(self.file, "w") as o:
            # Write the replay file
            json.dump(self.data, o)
