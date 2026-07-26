# Lux AI 2021 python game engine and Gymnasium
This is a replica of the Lux AI 2021 game ported directly over to python. It also provides a Gymnasium environment for training RL agents.


| **Features**                         | **LuxAi2021** |
| ------------------------------------ | ----------------------|
| Lux game engine porting to python    | :heavy_check_mark: |
| Documentation                        | :x: |
| All actions supported                | :heavy_check_mark: |
| PPO example training agent           | :heavy_check_mark:  |
| Example agent converges to a good policy | :heavy_check_mark: |
| Kaggle submission format agents      | :heavy_check_mark: |
| Lux replay viewer support            | :heavy_check_mark: |
| Game engine consistency validation to base game       | :heavy_check_mark: |

# Installation
This project uses Python 3.9.25 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest -q
```

Node.js is not required; seeded map generation runs entirely in Python.



# Python game interface
To directly use the ported game engine without the RL gym wrapper, here a couple example usages:

```
from luxai2021.game.game import Game
from luxai2021.game.actions import *
from luxai2021.game.constants import LuxMatchConfigs_Default


if __name__ == "__main__":
    # Create a game
    configs = LuxMatchConfigs_Default
    game = Game(configs)
    
    game_over = False
    while not game_over:
        print("Turn %i" % game.state["turn"])

        # Array of actions for both teams. Eg: MoveAction(team, unit_id, direction)
        actions = [] 

        game_over = game.run_turn_with_actions(actions)
    
    print("Game done, final map:")
    print(game.map.get_map_string())
```


# Python Gymnasium environment interface for RL

A Gymnasium interface and match controller was created that supports creating custom agents, and a framework to submit them in kaggle submissions. Keep in mind that this framework is built around one action per unit + city_tile that can act each turn. Creating a basic Gymnasium interface looks like the following, however you should look at the more complete example in the examples subfolder:

```
import random
from stable_baselines3 import PPO  # pip install stable-baselines3
from luxai2021.env.lux_env import LuxEnvironment, SaveReplayAndModelCallback
from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.game.game import Game
from luxai2021.game.actions import *
from luxai2021.game.constants import LuxMatchConfigs_Default
from functools import partial  # pip install functools
import numpy as np
from gymnasium import spaces
import time
import sys

class MyCustomAgent(AgentWithModel):
    def __init__(self, mode="train", model=None) -> None:
        """
        Implements an agent opponent
        """
        super().__init__(mode, model)
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actions_units = [
            partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            SpawnCityAction,
        ]
        self.actions_cities = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
        ]
        self.action_space = spaces.Discrete(max(len(self.actions_units), len(self.actions_cities)))
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,1), dtype=np.float16)

    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return
    
    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        return np.zeros((10,1))
    
    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            action_code: Index of action to take into the action array.
        Returns: An action.
        """
        # Map action_code index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y
            
            if city_tile != None:
                action =  self.actions_cities[action_code%len(self.actions_cities)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action =  self.actions_units[action_code%len(self.actions_units)](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            
            return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None
    
    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)
    
    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass
    
    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        if is_game_finished:
            if game.get_winning_team() == self.team:
                return 1 # Win!
            else:
                return -1 # Loss

        return 0
    

if __name__ == "__main__":
    # Create the two agents that will play eachother
    
    # Create a default opponent agent that does nothing
    opponent = Agent()
    
    # Create a RL agent in training mode
    player = MyCustomAgent(mode="train")
    
    # Create a game environment
    configs = LuxMatchConfigs_Default
    env = LuxEnvironment(configs=configs,
                     learning_agent=player,
                     opponent_agent=opponent)
    
    # Play 5 games
    env.reset()
    obs, _ = env.reset()
    game_count = 0
    while game_count < 5:
        # Take a random action
        action_code = random.sample(range(player.action_space.n), 1)[0]
        obs, reward, terminated, truncated, state = env.step(action_code)
        is_game_over = terminated or truncated
        
        if is_game_over:
            print(f"Game done turn {env.game.state['turn']}, final map:")
            print(env.game.map.get_map_string())
            obs, _ = env.reset()
            game_count += 1
    
    # Attach a ML model from stable_baselines3 and train a RL model
    model = PPO("MlpPolicy",
                    env,
                    verbose=1,
                    tensorboard_log="./lux_tensorboard/",
                    learning_rate=0.001,
                    gamma=0.998,
                    gae_lambda=0.95,
                    batch_size=2048,
                    n_steps=2048
                )
    
    print("Training model for 100K steps...")
    model.learn(total_timesteps=10000000)
    model.save(path='model.zip')

    # Inference the agent for 5 games
    game_count = 0
    obs, _ = env.reset()
    while game_count < 5:
        action_code, _states = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, state = env.step(action_code)
        is_game_over = terminated or truncated
        
        if is_game_over:
            print(f"Game done turn {env.game.state['turn']}, final map:")
            print(env.game.map.get_map_string())
            obs, _ = env.reset()
            game_count += 1



```

## Example python ML training
Create your own agent logic, observations, actions, and rewards by modifying this example:

https://github.com/glmcdona/LuxPythonEnvGym/blob/main/examples/agent_policy.py

Then train your model by:

```python ./examples/train.py```

You can then run tensorboard to monitor the training:

```tensorboard --logdir lux_tensorboard```


## Example kaggle notebook
Here is a complete training, inference, and kaggle submission example in Notebook format:

https://www.kaggle.com/glmcdona/lux-ai-deep-reinforcement-learning-ppo-example


## Preparing a kaggle submission

You have trained a model, and now you'd like to submit it as a kaggle submission. Here are the steps to prepare your submission.

Either view the above kaggle example or prepare a submission yourself:
1. Place your trained model file as `model.zip` and your agent file `agent_policy.py` in the `./kaggle_submissions/` folder.
1. Run `python download_dependencies.py` in `./kaggle_submissions/` to copy two required python package dependencies into this folder (luxai2021 and stable_baselines3).
1. Tarball the folder into a submission `tar -czf submission.tar.gz -C kaggle_submissions .`

**Important:** The model.zip needs to have been trained on Python 3.7.* or you get a deserialization error, since this is the python version that Kaggle Environment uses to inference the model in submission.

## Creating and viewing a replay
If you are using the example `train.py` to train your model, replays will be generated and saved along with a copy of the model every 100K steps. By default 5 replay matches will be saved with each model checkpoint into `.\\models\\model(runid)_(step_count)_(rand).json` to monitor your bot's behaviour. You can view the replay here:
https://2021vis.lux-ai.org/

Alternatively to manually generate a replay from a model, you can place your trained model file as `model.zip` and your agent file `agent_policy.py` in the `./kaggle_submissions/` folder. Then run a command like the following from that directory:

`lux-ai-2021 ./kaggle_submissions/main_lux-ai-2021.py ./kaggle_submissions/main_lux-ai-2021.py --maxtime 100000`

This will battle your agent against itself and produce a replay match. This requires the official `lux-ai-2021` to be installed, see instructions here:
https://github.com/Lux-AI-Challenge/Lux-Design-2021

## Behavior cloning from Kaggle replays

Train the shared Residual U-Net and factorized Worker, Cart, and City action heads from official Season 1
Kaggle replay JSON files. By default, only the winning player from each replay is used:

```bash
uv run python examples/train_bc.py \
  --replay-dir replay_datasets \
  --output-dir models/bc_v2 \
  --epochs 20
```

Use `--team-selection all --winner-weight 1.5` only when both players should be retained.
Training shows tqdm progress bars for class statistics, train, validation, and test phases; use `--no-progress`
for non-interactive logs.

Class counts are stored in `class_statistics.pt` and in every checkpoint. A resume with the same replay files,
split, team selection, turn limit, and schemas prints `Class statistics: checkpoint` and skips the dataset scan.
Use `--recompute-class-statistics` only after intentionally invalidating the cache.

Plot the train/validation loss and the per-head multiclass confusion matrices recorded in `metrics.json`:

```bash
uv run python examples/visualize_bc_metrics.py \
  --metrics models/bc_v2/metrics.json
```

Plots are saved under `models/bc_v2/plots/`. Confusion matrices include both row-normalized rates and raw
counts. Metrics created by an older trainer contain enough information for the loss curve only; run validation
with the updated trainer to record confusion matrices.

CUDA is selected automatically. The baseline training defaults are batch size 32, AMP, channels-last convolutions,
fused AdamW, cuDNN benchmarking, and up to four data-loader workers. If GPU memory is insufficient, reduce
`--batch-size`; if multiprocessing causes memory pressure, use `--num-workers 0`.

The version 2 hybrid observation uses 55 centered spatial channels. It keeps team-relative units, resources,
city night-survival information, coordinates, and board masks, while adding stacked-unit counts, cargo-full
flags, corrected cooldown normalization, and categorical embeddings for the 40-turn cycle, game phase, and
board size. The encoder masks padding at every U-Net stage and exposes pooled global features for a future
RL value head.

Training and inference share the same viable-action masks. Off-board moves, enemy-city moves, moves blocked by
cooldown units, impossible transfers, invalid city construction, exhausted research, and unit-cap production
are removed before softmax. Replay commands rejected by these masks are learned as their effective no-op result.

Resume while preserving the game-level train/validation/test split:

```bash
uv run python examples/train_bc.py \
  --replay-dir replay_datasets \
  --output-dir models/bc_v2 \
  --resume models/bc_v2/latest.pt \
  --epochs 40
```

Checkpoints produced with the earlier 44-channel feature schema cannot be resumed with this version.

### Team Durrett multi-source profile

The adapted Team Durrett profile keeps the 55-feature observation, factorized action heads, viable-action masks,
and pooled features for later RL use. It replaces the U-Net with a 384-channel, 18-block FReLU/SE encoder plus a
masked Transformer layer. The spatial trunk and per-action projections are shared, while the final classifier for
each action head is specific to the source submission.

The replay directory must contain `agent_info.json` above each source submission's replays and a matching
`*_info.json` beside every replay. The source player is selected by `submissionId`, independently of who won.
Source-versus-source self-play contributes both players to the same policy head.

```bash
uv run python examples/train_bc.py \
  --training-profile durrett \
  --replay-dir replay_datasets \
  --output-dir models/bc_durrett
```

The profile defaults to 100 epochs, batch size 16 with four-step gradient accumulation, AdamW at `1e-3`,
weight decay `1e-5`, and learning-rate drops at epochs 50 and 80. Worker/cart `stay` targets receive weight 0.3.
The source with the highest `lb` in `agent_info.json` is stored as the inference default. Per-source metrics and
the complete source catalog are written to the checkpoint and `metrics.json`.

Use one replay and one turn per source for a quick CUDA smoke:

```bash
uv run python examples/train_bc.py \
  --training-profile durrett \
  --replay-dir replay_datasets \
  --output-dir models/bc_durrett_smoke \
  --max-replays-per-source 1 \
  --max-turns 1 \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --device cuda \
  --num-workers 0
```

Resume with the saved split, scheduler, source mapping, optimizer, and cached class statistics:

```bash
uv run python examples/train_bc.py \
  --replay-dir replay_datasets \
  --output-dir models/bc_durrett \
  --resume models/bc_durrett/latest.pt
```

Use the best checkpoint in a local match:

```python
from luxai2021.imitation import BehaviorCloningAgent

agent = BehaviorCloningAgent("models/bc/best.pt", device="auto")
actions = agent.process_turn(game, team=0)
```

For a multi-source checkpoint, omit `source_id` to use the saved highest-`lb` source or select one explicitly:

```python
agent = BehaviorCloningAgent("models/bc_durrett/best.pt", device="auto", source_id=23692494)
```

The bundled replay fixtures are intended for smoke tests only. Use a larger replay collection for actual training.
