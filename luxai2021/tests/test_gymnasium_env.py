import numpy as np
from stable_baselines3.common.env_checker import check_env

from luxai2021.env.agent import Agent, AgentWithModel
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default


class NoOpLearningAgent(AgentWithModel):
    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        return None


def test_gymnasium_api_and_sb3_compatibility():
    env = LuxEnvironment(
        configs=LuxMatchConfigs_Default,
        learning_agent=NoOpLearningAgent(mode="train"),
        opponent_agent=Agent(),
    )

    check_env(env, warn=True)

    observation, info = env.reset(seed=123)
    assert observation.dtype == np.float16
    assert isinstance(info, dict)

    transition = env.step(env.action_space.sample())
    assert len(transition) == 5
