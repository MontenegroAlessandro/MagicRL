"""Half Cheetah Environment Implementation
Action Space
    Box(-1, 1, (6,), float32)
Observation Space
    Box(-inf, inf, (17,), float64)
"""
# Libraries
import gymnasium as gym
import copy
import numpy as np
from abc import ABC
from envs.base_env import BaseEnv


class HalfCheetah(BaseEnv, ABC):
    """Half Cheetah Wrapper for the environment by GYM."""
    def __init__(
            self, horizon: int = 0, gamma: float = 0.99, verbose: bool = False,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 0.1,
            reset_noise_scale: float = 0.1,
            exclude_current_positions_from_observation: bool = True,
            render: bool = False
    ) -> None:
        super().__init__(
            horizon=horizon,
            gamma=gamma,
            verbose=verbose
        )
        self.render = render
        render_mode = None
        if self.render:
            render_mode = "human"

        self.gym_env = gym.make(
            'HalfCheetah-v4',
            forward_reward_weight=forward_reward_weight,
            ctrl_cost_weight=ctrl_cost_weight,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            render_mode=render_mode
        )
        self.action_bounds = [-1, 1]
        self.state_dim = self.gym_env.observation_space.shape[0]    # 17
        self.action_dim = self.gym_env.action_space.shape[0]        # 6
        self.state = None
        return

    def step(self, action):
        obs, reward, done, _, _ = self.gym_env.step(action)
        self.state = copy.deepcopy(obs)
        return obs, reward, done, None

    def reset(self):
        obs = self.gym_env.reset()
        self.state = copy.deepcopy(obs[0])
        return obs

    def render(self, mode='human'):
        return self.gym_env.render()

    def close(self):
        self.gym_env.close()

    def sample_action(self):
        return self.gym_env.action_space.sample()

    def sample_state(self, args: dict = None):
        return self.gym_env.observation_space.sample()
