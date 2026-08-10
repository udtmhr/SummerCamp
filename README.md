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

Seven encoders can be trained with the same observation, action heads, and viable-action
masks:

- `unet` is the default 32x32 residual U-Net.
- `resnet17x32` is an RLIAYN-inspired full-resolution ResNet with 17 residual blocks and 32 channels.
  `resnet17x48` is its capacity-control variant with 48 channels. Both retain this project's observation,
  action heads, legal-action masks, and pooled RL value features; they are not exact reproductions of the
  unpublished RLIAYN implementation. Their flat distillation policies have 364,700 and 781,116 parameters,
  respectively.
- `resattn8` is the recommended compact hybrid: a 48/96/192-channel residual U-Net with two global SDPA
  blocks only at the 8x8 bottleneck. Its flat distillation policy has 4,409,500 parameters.
- `transformer16` applies an eight-layer Transformer to 16x16 tokens and decodes through a 32x32 skip.
- `axial32` keeps 32x32 features and alternates row and column attention in six axial blocks.
- `axial32_4m5` keeps the six axial blocks but uses 192-wide attention and a 672-wide FFN. The flat-policy
  distillation model has 4,505,692 parameters, versus 7,603,740 for the full `axial32` model.

CUDA training enables `torch.compile(..., mode="max-autotune")` and BF16 autocast by default. The lazy compilation warmup is
reported separately from epoch throughput and does not alter checkpoint keys; use `--no-compile` for an eager
baseline, `--compile-mode default|reduce-overhead|max-autotune` to select another mode, or
`--amp-dtype float16` to explicitly request FP16. Axial attention uses PyTorch scaled-dot-product attention
kernel dispatch while retaining the learned relative-distance bias and padding mask.
Training stops before an optimizer update when a non-finite loss or gradient is detected.

Use the same seed, effective batch, and shared class-statistics cache for an architecture comparison:

Replay JSON is parsed only while building a binary cache. Reuse the same `--replay-cache-dir` across runs;
unchanged replays are loaded from the cache in later epochs and later architecture runs. Source size and
modification time automatically invalidate stale entries. Training DataLoaders use pinned host memory and
non-blocking device transfers.

```bash
uv run python examples/train_bc.py \
  --encoder-type unet \
  --replay-dir replay_datasets \
  --output-dir models/bc_encoder_compare/unet \
  --class-statistics-path models/bc_encoder_compare/class_statistics.pt \
  --replay-cache-dir models/bc_encoder_compare/replay_cache \
  --batch-size 8 \
  --gradient-accumulation-steps 4

uv run python examples/train_bc.py \
  --encoder-type transformer16 \
  --replay-dir replay_datasets \
  --output-dir models/bc_encoder_compare/transformer16 \
  --class-statistics-path models/bc_encoder_compare/class_statistics.pt \
  --replay-cache-dir models/bc_encoder_compare/replay_cache \
  --batch-size 8 \
  --gradient-accumulation-steps 4

uv run python examples/train_bc.py \
  --encoder-type axial32 \
  --replay-dir replay_datasets \
  --output-dir models/bc_encoder_compare/axial32 \
  --class-statistics-path models/bc_encoder_compare/class_statistics.pt \
  --replay-cache-dir models/bc_encoder_compare/replay_cache \
  --batch-size 8 \
  --gradient-accumulation-steps 4
```

Compare best validation loss, active-action accuracy, training throughput, peak CUDA memory, and parameter
counts:

```bash
uv run python examples/compare_bc_architectures.py \
  --run unet=models/bc_encoder_compare/unet/metrics.json \
  --run transformer16=models/bc_encoder_compare/transformer16/metrics.json \
  --run axial32=models/bc_encoder_compare/axial32/metrics.json \
  --output-dir models/bc_encoder_compare/comparison
```

The comparison rejects runs with different splits, class-statistics signatures, or training settings. It writes
`architecture_comparison.json`, `validation_loss_comparison.png`, and `resource_comparison.png`.

Evaluate selected best checkpoints on the exact same test split in FP32:

```bash
uv run python examples/evaluate_bc_checkpoints.py \
  --checkpoint unet=models/bc_v2/best.pt \
  --checkpoint transformer16=models/bc_encoder_compare/transformer16/best.pt \
  --checkpoint axial32=models/bc_encoder_compare/axial32/best.pt \
  --output models/bc_encoder_compare/evaluation.json \
  --device cuda
```

The evaluator rejects non-finite checkpoints and mismatched splits or class-statistics signatures. Its throughput
numbers use eager FP32 execution so that loss comparison is independent of AMP and compilation settings.
By default, `--match-seeds 0` preserves the test-loss-only evaluation.

After the test-split evaluation, run a parallel round robin with 50 seeds and both team assignments:

```bash
uv run python examples/evaluate_bc_checkpoints.py \
  --checkpoint unet=models/bc_v2/best.pt \
  --checkpoint transformer16=models/bc_encoder_compare/transformer16/best.pt \
  --checkpoint axial32=models/bc_encoder_compare/axial32/best.pt \
  --output models/bc_encoder_compare/evaluation.json \
  --device cuda \
  --match-seeds 50 \
  --match-workers auto
```

On CUDA, `--match-workers auto` uses two spawned processes sharing the GPU; on CPU it uses at most four, and
the worker count is always capped by the number of seeds. Three checkpoints produce 300 games
(`3 pairs x 50 seeds x 2 team assignments`). Replays are not written. With matches enabled, `winner` is ranked
by round-robin score rate while `test_winner` preserves the test-loss winner.

For a short CUDA smoke, replace `replay_datasets` with
`luxai2021/tests/replays_for_test/27095556.json`, add `--max-turns 8 --epochs 1 --device cuda`, and run each
encoder command above. Resume a run with its existing encoder:

```bash
uv run python examples/train_bc.py \
  --replay-dir replay_datasets \
  --output-dir models/bc_encoder_compare/axial32 \
  --resume models/bc_encoder_compare/axial32/latest.pt \
  --epochs 40
```

GradScaler state is saved and restored for FP16 runs. Older checkpoints without scaler state remain loadable.
A checkpoint containing non-finite model parameters is rejected; recover a failed Transformer run from its
finite `best.pt` checkpoint using the default BF16 mode:

```bash
uv run python examples/train_bc.py \
  --replay-dir replay_datasets \
  --output-dir models/bc_encoder_compare/axial32 \
  --resume models/bc_encoder_compare/axial32/best.pt \
  --epochs 20 \
  --amp-dtype bfloat16
```

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

Use the best checkpoint in a local match:

```python
from luxai2021.imitation import BehaviorCloningAgent

agent = BehaviorCloningAgent("models/bc/best.pt", device="auto")
actions = agent.process_turn(game, team=0)
```

The bundled replay fixtures are intended for smoke tests only. Use a larger replay collection for actual training.

### Distill the Lux AI 2021 first-place policy

The distillation pipeline ports the first-place 24-block policy and its exact 19/17/4 flat action ordering without
adding the upstream package as a dependency. It uses offline distillation: the teacher logits, including the original
180-degree rotation ensemble, are computed once and reused by every student and epoch. For the fixed replay states,
this preserves the targets that would be recomputed online while avoiding repeated teacher inference.

Download and SHA-256 verify the upstream MIT-licensed checkpoint, then cache logits for both players in every replay:

```bash
uv run python examples/download_first_place_teacher.py

# Optional: add 100 full teacher-vs-teacher games at seeds 10000..10099.
# The checkpoint is loaded once and shared by both sides; auto enables the
# first-place 180-degree rotation ensemble. Re-running skips completed seeds.
uv run --locked python examples/generate_first_place_selfplay.py \
  --teacher-checkpoint models/teachers/lux_2021_first_place/062179520_weights.pt \
  --output-dir replay_datasets/first_place_selfplay \
  --games 100 \
  --seed-start 10000 \
  --device cuda \
  --tta auto

uv run python examples/precompute_first_place_targets.py \
  --replay-dir replay_datasets \
  --teacher-checkpoint models/teachers/lux_2021_first_place/062179520_weights.pt \
  --output-dir models/teachers/lux_2021_first_place/cache \
  --replay-cache-dir models/replay_cache \
  --device cuda
```

Precompute the compact student inputs once. This moves replay parsing, board encoding, legal-mask construction, and
target construction out of every training epoch. FP16 observations are converted back to FP32 before the model and
are suitable for the default BF16 training; use `--observation-dtype float32` for bit-preserving inputs.

```bash
uv run python examples/precompute_distillation_dataset.py \
  --replay-dir replay_datasets \
  --teacher-cache-dir models/teachers/lux_2021_first_place/cache \
  --replay-cache-dir models/replay_cache \
  --output-dir models/teachers/lux_2021_first_place/prepared \
  --observation-dtype float16 \
  --num-workers 4
```

Train the recommended attention hybrid against the shared prepared cache:

```bash
uv run --locked python examples/train_distilled_bc.py \
  --encoder-type resattn8 \
  --replay-dir replay_datasets \
  --teacher-cache-dir models/teachers/lux_2021_first_place/cache \
  --prepared-cache-dir models/teachers/lux_2021_first_place/prepared \
  --output-dir models/distilled/resattn8 \
  --batch-size 64 \
  --num-workers 8 \
  --prefetch-factor 2 \
  --amp-dtype bfloat16 \
  --device cuda
```

For an architecture-only comparison, train the two RLIAYN-inspired widths against the same prepared cache and
training settings. Fresh `resnet17x32` and `resnet17x48` runs intentionally start directly from distillation:

```bash
for encoder in resnet17x32 resnet17x48; do
  uv run --locked python examples/train_distilled_bc.py \
    --encoder-type "$encoder" \
    --replay-dir replay_datasets \
    --teacher-cache-dir models/teachers/lux_2021_first_place/cache \
    --prepared-cache-dir models/teachers/lux_2021_first_place/prepared \
    --output-dir "models/distilled/$encoder" \
    --batch-size 64 \
    --num-workers 8 \
    --prefetch-factor 2 \
    --amp-dtype bfloat16 \
    --seed 42 \
    --device cuda
done
```

Keep the replay split, seed, effective batch size, epoch count, teacher cache, and augmentation fixed when comparing
these runs with `resattn8`. Parameter count, best validation metrics, throughput, and peak memory can be summarized
with `examples/compare_bc_architectures.py`; strength still requires matched-seed games with swapped teams.

`resattn8` starts directly from distillation unless `--student-checkpoint` supplies a matching pretrained
checkpoint. The full Transformer encoders remain loadable for existing checkpoint compatibility but are no longer
recommended for new distillation runs. Training combines temperature-scaled KL loss with replay hard labels. CUDA
training enables BF16, TF32,
channels-last tensors, fused AdamW, variable-size tail batches without dummy padding, and replay caching. The legacy
`--no-compile` flag is accepted as a no-op for command compatibility. D4 augmentation runs as one vectorized GPU
operation, entity targets are padded only to the largest real entity count in each batch, and metric reductions
synchronize once per epoch instead of once per batch. Distilled checkpoints record
`inference_augmentation=rot180`, so local play enables the batched
two-view ensemble automatically. Override it for latency comparisons with `--tta-a none` or `--tta-b none`:

Distillation uses validation-driven learning-rate decay by default. It keeps the current rate until validation loss
fails to improve by at least `0.002` for three validations, then multiplies it by `0.5`, down to `5e-6`. Scheduler
state and the learning rate used by each epoch are saved in checkpoints and `metrics.json`. Resume
`resattn8_v2` for ten more epochs while preserving its teacher-only objective and current `1e-4` rate with:

```bash
uv run --locked python examples/train_distilled_bc.py \
  --encoder-type resattn8 \
  --replay-dir replay_datasets \
  --teacher-cache-dir models/teachers/lux_2021_first_place/cache \
  --prepared-cache-dir models/teachers/lux_2021_first_place/prepared \
  --output-dir models/distilled/resattn8_v2 \
  --resume models/distilled/resattn8_v2/best.pt \
  --epochs 30 \
  --batch-size 64 \
  --num-workers 8 \
  --prefetch-factor 1 \
  --distill-weight 1.0 \
  --hard-label-weight 0.0 \
  --lr-scheduler plateau \
  --lr-decay-factor 0.5 \
  --lr-patience 2 \
  --lr-threshold 0.002 \
  --min-learning-rate 0.000005 \
  --amp-dtype bfloat16 \
  --device cuda
```

On resume, omitting `--learning-rate` preserves the checkpoint rate and scheduler progress. Supplying it explicitly
starts the resumed optimizer from that rate. Use `--lr-scheduler none` only for a fixed-rate comparison.

```bash
uv run python main.py \
  --model-a models/distilled/resattn8/best.pt \
  --model-b models/distilled/unet/best.pt \
  --tta-a auto --tta-b auto --device cuda
```

The teacher adapter is pinned to upstream commit `973a6c6c63211b6c7ab6fdf50e026e458d1f6e4e` and checkpoint SHA-256
`40248f0fbc9b8e1e1b1f7cc6fc674c041d8dac43b964ae45bd976d927cdffd22`. See
[`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) for attribution.

### Codex-guided evolutionary reinforcement learning

`examples/evolve_rl.py` evolves bounded reward programs, PPO parameters, and opponent mixtures while keeping the
distilled UNet/ResAttn8 architectures and `first_place_flat_v1` action schema fixed. Codex proposes structured JSON
only; proposals are validated before training and cannot execute arbitrary reward code. New runs use a shared
`joint_sequential_v2` decoder for rollout and evaluation, rot180 policy/value TTA, turn-level joint PPO ratios,
online first-place Teacher KL, and the prepared replay cache as an auxiliary BC anchor. Exported `best.pt` files
remain compatible with `BehaviorCloningAgent`.

New runs embed the official Lux AI Season 1 rules URL and the SHA-256 of the packaged rule summary in the schema-v5
manifest and every Codex prompt. The manifest fixes reward metric schema v3 so a run cannot mix candidates evaluated
with different metric sets. The search treats the original first-place Teacher as the strongest anchor: Teacher
score is ranked before aggregate score, and final promotion requires paired Teacher non-regression as well as a
non-negative distilled-base delta. The optional `teacher_guarded_near_sparse` curriculum raises the Teacher opponent
floor from 25% to 50%, keeps at least 10% snapshot play, and anneals potential-based shaping from 100% to 5% while
leaving the terminal win/loss reward unchanged. Use `--curriculum-profile terminal_only_ablation` for an explicit
terminal-only ablation or `legacy` only when reproducing an old run.

The default dense-shaping reward uses linear potential differencing with absolute own-team penalties for next-night
fuel deficit and cumulative city-tile loss. Raw terminal win/loss is +/-10 and the complete reward is divided by 10.
With reward scale 0.35 and potential clipped to [-5, 5], normalized shaping is strictly below 0.35. Fixed normalization
is used instead of running return normalization to keep reward meaning and resumes deterministic. Reward and PPO gamma
must match and default to 0.999; TD(lambda) uses lambda 0.995 instead of adding UPGO. The BC anchor coefficient is 0.05
and its multiplier stays at 1.0 through 20% progress, then reaches 0.8 at 70% and 0.2 at completion. Dense shaping is
also annealed from 1.0 at 20% to 0.5 at 70% and 0.25 at completion without changing the terminal reward.
Reward Program v3 sets terminal-state potential to zero, including draws, so discounted shaping leaves no terminal
optimization target besides match outcome.
Because candidates and curricula are content-frozen in run artifacts, use a new run directory when adopting these
defaults; an older run with a different stored curriculum is rejected rather than silently changing its learning rule.

The default BC anchor uses 128 deterministically selected replays, all available turns, and phase-balanced batches
over four 90-turn phases crossed with day/night. Final stages export curriculum milestones and screen them against the
Teacher before the selected checkpoint enters the full league evaluation. League reports include city extinction,
last-night survival, normalized city loss, per-city night fuel margin, and the maximum stranded-fuel fraction observed
at night starts. Guarded promotion allows at most a +0.02 paired mean regression in stranded fuel.

Each Codex feedback record includes the exact turns where city tiles disappeared from night fuel shortage, fuel and
upkeep at destruction, the worst night-start stranded-fuel fraction and turn, illegal actions rejected by the engine,
the candidate's setting changes from its parent, and the resulting score/teacher-score deltas. Dynamic action
restrictions are applied only by intersecting them with
the existing legal-action mask, so an existing forbidden action can never be re-enabled by evolution.

The four islands have fixed roles: `i00` makes one or two coefficient changes, `i01` makes one local AST edit,
`i02` adds or removes one direct normalized metric, and `i03` performs a free bounded reward redesign, crossover,
or restart. `feature_generated` remains readable in older candidate files but is never proposed for new candidates.
Non-restart candidates inherit the primary parent's policy checkpoint; only `i00` also inherits its value head.
Value-head resets run a critic-only warm-up before PPO. Parent-policy KL loss and L2-SP are disabled: PPO clipping,
the first-place Teacher distillation anchor, actor learning-rate ramp, and staged evaluation provide stability without
forcing candidates to remain near a weak distilled parent. The illegal-action auxiliary loss acts on unmasked logits
while sampling still uses the hard mask.
Reward expressions remain bounded structured data rather than arbitrary Python code.

On `i03`, Codex may coordinate changes across multiple reward components and subtrees. The server derives the effective
mutation class from the actual parent/child diff instead of trusting the declared island contract. Approximate AST
distance 0.20 to 0.65 is an advisory diversity target: smaller safe edits are reclassified as parameter, feature, or
local structural changes, while larger safe edits remain policy-inheriting large structural candidates and face the
normal short-stage screen. Only unsafe Reward IR, invalid lineage, non-finite values, resource-limit violations, and
mask relaxation are rejected. Without Codex, the normal fallback mix is 50% structural, 30% crossover, and 20% restart. After two stagnant generations it
becomes 40%, 20%, and 40%, respectively; unavailable crossover probability is reassigned to structural exploration.

Metric schema v3 adds the relative count of Cities funded through the next night (`safe_city_tiles`) and a cumulative
city-loss signal that remains linear through 64 lost tiles (`own_city_tiles_lost_linear`). Schema-v2 candidate and
checkpoint files remain readable, but a schema-v2 run cannot be resumed as schema v3. The Metric DSL also exposes
phase-aware and local survival signals: normalized turns until night/night turns
remaining, minimum per-city survival, at-risk city-tile fraction, next-night fuel deficit, nearby cargo coverage,
avoidable stranded fuel across disconnected Cities, cumulative night city loss, worker resource access/cargo
fullness, unit-capacity utilization, and coal/uranium unlock state. `own_stranded_fuel` is normalized to `[0, 1]` and
is nonzero only when one City has fuel surplus while another has a simultaneous deficit; it is available to reward
search but is not added to the default reward. Selectors include at-risk/safe city tiles,
fuel-carrying/full/actionable workers, and resources the
team can currently harvest. Cumulative loss metrics are intentional: potential differencing avoids the large
opposite-sign reward that a one-turn event pulse would produce when it resets to zero. Exact action-history signals
such as oscillation are not exposed until they can distinguish deliberate cooldown waits from genuine repeated-action
failure.

First verify the 24-candidate schedule without training:

```bash
uv run --locked python examples/evolve_rl.py \
  --run-dir models/evolution/lux-s1-001-dry-run \
  --dry-run
```

Then launch the default staged run on one GPU:

```bash
uv run --locked python examples/evolve_rl.py \
  --run-dir models/evolution/lux-s1-001 \
  --device cuda \
  --resattn8-only
```

Run the matched Lux Survival v2 A/B experiment and advance only a candidate that passes the paired short-stage gate:

```bash
uv run --locked python examples/evolve_rl.py \
  --run-dir shared/lux-survival-v2-ab \
  --fixed-candidate configs/rl_candidates/survival_credit_v2.json \
  --fixed-candidate configs/rl_candidates/survival_safe_city_v2.json \
  --resattn8-checkpoint models/distilled/resnet17x48_s42/best.pt \
  --device cuda --resattn8-only --curriculum-profile dense_shaping \
  --rollout-backend threaded \
  --rollout-envs 8 --budget-unit games \
  --short-games 384 --medium-games 1536 --final-games 6144 \
  --episodes-per-update 64 --rollout-compile off \
  --screening-seeds 24 --medium-count 1 --final-count 1
```

Repeated `--fixed-candidate` values are content-frozen together in the manifest. Short selection allows at most a 0.02
paired score/survival regression; medium selection requires non-negative overall, base, and Teacher deltas. If no
candidate passes, the run stops before allocating the next-stage training budget.

To train the reviewed survival-focused candidate only, through short, medium, and final stages, use:

```bash
uv run --locked python examples/evolve_rl.py \
  --run-dir shared/lux-survival-linear-v1 \
  --fixed-candidate configs/rl_candidates/survival_linear_v1.json \
  --resattn8-checkpoint models/distilled/resnet17x48_s42/best.pt \
  --curriculum-profile dense_shaping \
  --rollout-backend threaded --rollout-compile off \
  --device cuda --resattn8-only
```

Fixed-candidate mode forces a zero-generation run. One path preserves the legacy single-candidate behavior; repeated
paths form a matched candidate set and only candidates passing the stage gate can advance. Source JSON files and
digests are frozen in the run manifest, so use a new run directory if any candidate definition changes.

PPO's existing GAE return is the TD(lambda) value target. Survival v2 uses lambda 0.995 with gamma 0.999 to extend
credit assignment in 360-turn games. UPGO is intentionally not mixed into the single shaped-reward
critic: AlphaStar applies it to a separate win/loss baseline, while using it here would self-imitate positive shaping
trajectories and could reinforce the City-expansion exploit. Add a separate terminal-only critic before enabling UPGO.

`--resattn8-only` disables UNet opponents, probes, training, baselines, and final evaluation. New runs default to
384/1,536/6,144 completed games for short/medium/final, with 64 games per PPO update. New runs hold the actor at the
candidate learning rate (1e-6 by default); joint-drift feedback may reduce it as low as 10% of that cap.
Stable per-action KL above 0.0003 or an action clip fraction above 0.08 halves the feedback multiplier for later
updates. Absolute joint KL above 0.01 or turn-joint log-ratio p95 above 0.20 also halves it. Distilled-base reference
KL is diagnostic-only and never changes LR or stops training. The default PPO
early-stop target is 0.00025.
PPO early stopping uses that non-negative per-action KL averaged over a complete epoch, rather than a single noisy
joint-turn minibatch. Effective actor learning rate, completed/planned minibatches, and any stop reason are recorded
in every update checkpoint. Diagnostic update selection records action agreement on a fixed
96-state phase-balanced base probe; among the remaining checkpoints, Teacher scores within one game are treated as
tied and overall score breaks the tie. Use `--actor-lr-schedule constant` only for a controlled ablation. Use
`--budget-unit decisions` with the legacy decision flags only for old manifests. A candidate runs
8 Lux environments concurrently by default. `--rollout-backend lockstep` selects a rendezvous: candidate and opponent
requests wait for all active games and then run as one fixed-size GPU batch. Games that finish early leave the rendezvous before
the next turn. The default `auto` currently retains `threaded` because the end-to-end 2x acceptance gate has not been
met; the effective backend and fallback reason are recorded in artifacts.
`--rollout-precision auto` uses BF16 on supported CUDA devices and FP32 elsewhere. `--rollout-compile off` is the
default. Joint runs require rot180 TTA, so actor, base, Teacher, and snapshot inference remain eager even if compile
is requested. Compile attempts, fallback details, and the effective mode are recorded and should remain zero/off.
PPO evaluates one joint action ratio per Lux turn. Tune `--rollout-envs` for each GPU and
`--episodes-per-update` for the rollout/update balance. Metrics record batch fill, queue wait, tensor staging,
forward/copy time, game step, observation encoding,
action decoding, reward metrics, rollout throughput, PPO update time, precision, compilation, fallback reason, and
peak CUDA allocation. PPO metrics also report `bc_anchor_seconds`. Phase-balanced BC retains the decompressed
prepared caches for all selected anchor replays; the default 128-replay run uses about 10 GiB of additional host
memory to avoid repeatedly decompressing an entire replay for each minibatch. Each candidate artifact also stores
the calibrated backend, precision, compile state, and
fallback reason in `rollout_runtime.json`. In ResAttn8-only mode, screening/medium/final evaluation uses 12/24/100
fixed seeds, both player orientations, the distilled ResAttn8 base plus the original first-place policy, no replay
output, paired bootstrap reporting, and an inference-latency guard. Re-running the same command resumes completed
candidates and stage checkpoints. Use
`--no-codex` for deterministic numeric mutations, or `--overwrite-run` to intentionally start that run directory
again. A dry-run directory cannot be reused for training because its candidates were generated without Codex. The
Codex CLI must already be installed and authenticated when proposal generation is enabled. Prompts are supplied via
stdin (`codex exec ... -`) instead of argv and stored as gzip audit artifacts. Feedback is limited to the parents'
highest completed stages, any higher-stage parent failures, and two distinct top candidates; diagnostic turns are
stored as lossless ranges. Semantic contract deviations are normalized without a retry. Only malformed, unsafe, or
lineage-invalid proposals are returned to Codex with the exact validator error and retried twice by default
(`--codex-validation-retries`). Rejected proposals and errors are retained beside the accepted proposal for
audit. After repair retries are exhausted, proposal errors stop the coordinator before registering or training that
candidate; deterministic fallback is available only with the explicit `--allow-codex-fallback` option.

Benchmark the target GPU before a long run. The first configuration is the speedup baseline; additional checkpoints
compare the existing lightweight students without replacing any model:

```bash
uv run --locked python benchmarks/benchmark_evolve_rollout.py \
  --device cuda --decisions 40000 --repeats 3 \
  --checkpoint resattn8=models/distilled/resattn8_v2_selfplay_ft/best.pt \
  --checkpoint resnet17x32=models/distilled/resnet17x32_s42/best.pt \
  --checkpoint resnet17x48=models/distilled/resnet17x48_s42/best.pt \
  --rollout-envs 1 --rollout-envs 2 --rollout-envs 4 --rollout-envs 8 \
  --backend lockstep --precision auto --compile auto \
  --output /tmp/lux-rollout-benchmark.json
```

Use `--seconds 60` instead of `--decisions 40000` for the fixed-wall-clock comparison. The budget flags are mutually
exclusive; omitting both uses 40,000 decisions.

Throughput alone does not promote a lightweight model. Train it in a separate run and apply the existing matched
seed/orientation, Teacher non-regression, and city/fuel survival gates. Retain it only when those gates pass and median
rollout throughput is at least 25% above the ResAttn8 control.

#### Smartphone notifications

The coordinator can notify run start, each completed generation, medium/final stage transitions, expired worker leases,
final completion, failures, and Ctrl+C. Delivery is best-effort and never changes a job result. Delivered event IDs are
stored in `<run-dir>/notifications/sent.json`, so resuming a run does not resend old progress events. Configuration is
read only from environment variables and is never written to the run manifest or notification state. Install the ntfy
phone app, subscribe to a private random topic, and configure PC1 only:

```bash
export LUX_EVOLUTION_NTFY_TOPIC='replace-with-a-long-random-topic'
# Optional for an authenticated/self-hosted server:
export LUX_EVOLUTION_NTFY_SERVER='https://ntfy.example.com'
export LUX_EVOLUTION_NTFY_TOKEN='tk_...'

uv run --locked python examples/test_evolution_notification.py \
  --run-dir /home/ueda/workspace/LuxPythonEnvGym/shared/lux-evolution-v3
```

The public `ntfy.sh` service has open topics by default, so use an unguessable topic and do not include sensitive run
data, or use an authenticated self-hosted server. Workers on ws3 need no notification configuration because the PC1
coordinator sends run-level notifications.

Run a short end-to-end check before committing a long GPU allocation:

```bash
uv run --locked python examples/evolve_rl.py \
  --run-dir /tmp/lux-evolution-smoke \
  --no-codex --device cpu --overwrite-run \
  --islands 1 --initial-per-island 1 --generations 0 \
  --short-seconds 0 --medium-count 0 --final-count 0 \
  --episodes-per-update 1 --screening-seeds 1 --max-turns 4
```

Candidate definitions, failures, training checkpoints, league games, and the promotion report are written under the
run directory so an interrupted search remains inspectable and resumable. Before the first checkpoint,
`training_progress.json` reports initialization, reward calibration, critic warm-up, rollout waves, and PPO update
phases; the same heartbeat is printed to the terminal. `latest_rl.pt` uses checkpoint schema v6 and
stores cumulative decisions, turns, episodes, optimizer/RNG state, curriculum progress, LR feedback state, history,
and BC sampler position.
Games-budget stages use one PPO epoch with 256-turn minibatches by default. The stage-entry policy is evaluated first;
after every update, the same fixed base/Teacher seeds and stage-entry action probe are evaluated again. Base action
agreement and reference KL are diagnostics only; they never change LR, reject an update, or stop a stage. An update
is rejected when either joint-policy drift guard is exceeded or matched Teacher/overall win-loss scores regress beyond
the configured one-/two-win margins.
Rejection restores `latest_rl.pt` from `accepted_rl.pt`, preserves the rejected diagnostic under
`diagnostic_evaluations/`, and ends the stage; only accepted checkpoints can become `best.pt`/`best_rl.pt` or feed the
next rollout. Start these settings in a new run directory because they are part of the immutable training contract.
Use `configs/rl_candidates/terminal_only_v1.json` as a fixed control candidate when separating PPO/terminal credit
assignment from dense-shaping misspecification; its shaping scale is exactly zero.
Use `configs/rl_candidates/survival_safe_city_joint_guarded_v1.json` for the conservative actionwise-PPO variant:
per-action clip 0.03, actor LR capped at 1e-6, and one PPO epoch. Run it with
`--actor-lr-schedule constant --actor-lr-warmup-updates 0 --actor-lr-floor-ratio 0.1`; this holds at 1e-6 unless a
drift guard reduces it (down to 1e-7). Joint KL above 0.01 or
turn-joint log-ratio p95 above 0.20 rolls the update back and stops the stage, while the existing matched base/Teacher
evaluation continues to detect performance regression.
PPO clips each decoded entity action and each non-trivial sampled conflict-priority step independently while sharing
the turn Advantage. Their log-probability sum still reconstructs the recorded turn-joint log probability.
The surrogate averages all valid clipped action/priority factors in the minibatch. This treats each factor as one PPO
sample and prevents the loss scale from growing with the factor count, while turns with more valid factors contribute
more samples. `action_clip_fraction` drives LR feedback; `joint_clip_fraction` remains diagnostic for comparison with
older runs. `joint_loss_reference_actions` now normalizes only the entropy, Teacher/reference KL, and illegal-action
auxiliary terms; it does not scale the actionwise policy surrogate.
Re-running a stage resumes its remaining decision budget. Each Teacher-selected inference `best.pt` has a matching
`best_rl.pt`; short-to-medium and medium-to-final inheritance loads its policy and value head, resets optimizer/RNG and
stage counters, then calibrates the critic before deciding whether warm-up is needed. Older runs without `best_rl.pt`
fall back to policy-only `best.pt`. Schema-v5 and older training checkpoints remain usable for policy/value stage
inheritance, but cannot exact-resume optimizer state because the policy and value optimizer groups changed.

The first coordinator invocation owns the schema-v12 run manifest: later invocations reuse its checkpoint paths,
SHA-256 descriptors, Codex/deterministic generation mode, model, and training budgets instead of overwriting them with
new CLI defaults. Runs without metric schema v3 remain untouched but must use a new run directory after this metric
change. Every candidate has a separate immutable record under `provenance/`; accepted Codex outputs retain
the raw `*.raw.json`, canonical `codex-gXX-iXX.json`, compressed prompt, and `*.meta.json` hashes plus the normalization
report. Medium and final candidate IDs are frozen under `selections/` when
each stage starts, and final selection is rejected unless every candidate has a completed medium result. Failed result
markers are archived and retried, while completed results remain immutable. Same-candidate resume may tolerate a
different textual base-model path only when available SHA-256 lineage remains identical. CUDA RNG is restored only for
the active training device, so a checkpoint remains resumable when the source and destination workers expose different
GPU counts. Fatal CUDA context errors such as `unspecified launch failure` stop
the worker after releasing its lease. Restart that worker process to recreate the CUDA context; the job is then claimed
again from its last checkpoint. A valid terminal `summary.json` has `status: "completed"`, records both frozen selection
lists, and requires `integrity.valid_for_promotion: true` before any candidate can be promoted.

Independent candidates can use another GPU machine without a shared filesystem. The coordinator keeps the authoritative
queue and artifacts locally and exposes a loopback-only Job API. Start it on PC1; omitting `--coordinator-only` lets PC1
also train candidates while serving ws3:

Worker-local dependency failures such as a missing BC-anchor replay or a reset CUDA context are requeued up to two
times. If the same infrastructure failure persists, that candidate/stage is recorded under `jobs/skipped/`; the wave
continues when another candidate completed successfully, but stops when every candidate failed because no fair ranking
is possible. Reward/DSL errors are not treated as infrastructure failures.

```bash
export LUX_EVOLUTION_JOB_TOKEN='<same-random-token-on-both-machines>'
uv run --locked python examples/evolve_rl.py \
  --run-dir /home/ueda/workspace/LuxPythonEnvGym/shared/lux-evolution \
  --device cuda --resattn8-only --distributed --job-api-listen 127.0.0.1:8765
```

When ws3 cannot authenticate to lyra with a forwarded agent, relay the API through the Mac that owns the working SSH
configuration. Do not copy private keys into ws3. In Mac terminal A, forward the PC1 API to the Mac:

```bash
ssh -N -L 127.0.0.1:28765:127.0.0.1:8765 \
  -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  ueda
```

In Mac terminal B, expose that Mac-local port only on ws3's loopback interface:

```bash
ssh -N -R 127.0.0.1:18765:127.0.0.1:28765 \
  -o ServerAliveInterval=60 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
  tomohiro_ws3
```

Keep both Mac SSH processes running. In a ws3 terminal, verify the forwarded endpoint before starting Docker:

```bash
curl http://127.0.0.1:18765/healthz
```

It should return `{"status":"ok","api_version":2}`. `Connection refused` means the SSH tunnel is not running on
ws3; checking PC1's port `8765` does not verify ws3's forwarded port `18765`.

Start the ws3 Docker worker in another terminal. Host networking lets the container reach the loopback SSH tunnel;
the worker run directory is local persistent scratch, not a shared mount:

```bash
mkdir -p "$HOME/lux-evolution-worker"
docker run --rm --gpus all --network host \
  -e LUX_EVOLUTION_JOB_TOKEN \
  -e PYTHONUNBUFFERED=1 \
  -e UV_PROJECT_ENVIRONMENT=/tmp/lux-evolution-venv \
  -e UV_LINK_MODE=copy \
  -v "$HOME/lux-evolution-worker:/workspace/lux-evolution-worker" \
  -v "$PWD:/app" -w /app \
  ueda/uv-app \
  uv run --locked python examples/evolve_rl.py \
    --run-dir /workspace/lux-evolution-worker \
    --device cuda --worker --worker-id ws3-gpu0 \
    --job-api-url http://127.0.0.1:18765 \
    --worker-idle-seconds 0
```

The repository revision and base model files must exist in the image on both machines. Checkpoint/cache path flags are
local to each machine, so pass ws3-specific paths after the image command when needed. The API sends candidate context
to preserve parent-change feedback, returns completed checkpoints and diagnostics to PC1, and streams parent/resume
checkpoints with SHA-256 descriptors. Workers cache downloads by content hash. The v1 claim/upload routes remain
available for older workers. Completion uploads are idempotent, while stale claims older
than 12 hours are requeued. Every 10 minutes by default, ws3 uploads `latest_rl.pt` with a lease heartbeat; a restarted
worker downloads that partial stage and continues its remaining decision budget. Ctrl-C releases the lease after a
best-effort final checkpoint upload. Change the transfer interval with `--job-heartbeat-seconds`. Set
`LUX_EVOLUTION_JOB_TOKEN` to the same random value on PC1 and ws3; it is not written to
the run manifest. `--worker-idle-seconds 0` keeps the remote worker alive across generation barriers. Add
`--coordinator-only` only when PC1 should schedule without using its own GPU.

### Play against the original first-place model

Use the normal one-on-one CLI with a BC or distilled checkpoint on one side and the original first-place teacher on
the other:

```bash
uv run python main.py \
  --model-a models/bc_v2/best.pt \
  --type-a bc \
  --model-b models/teachers/lux_2021_first_place/062179520_weights.pt \
  --type-b first-place \
  --tta-b auto \
  --seed random \
  --device auto
```

Set either `--type-a` or `--type-b` to `first-place` for the upstream teacher checkpoint. Ordinary BC and distilled
checkpoints use `bc`. The first-place model enables its original 180-degree ensemble when TTA is `auto`. The replay
uses the same timestamp-and-winner filename convention as other one-on-one matches.

Run repeated one-on-one games and report each model's win rate with `--games`. The first game uses `--seed` and later
games use consecutive seeds; omit `--seed` or use `--seed random` to choose the first seed randomly. Replays are
disabled by default:

```bash
uv run python main.py \
  --model-a models/bc_v2/best.pt --type-a bc \
  --model-b models/distilled/resattn8_v2/best.pt --type-b bc \
  --games 20 --seed random --device auto
```

Add `--save-replays --replay-dir replays/comparison` to save every game's replay. Without `--save-replays`, no replay
files are written.
