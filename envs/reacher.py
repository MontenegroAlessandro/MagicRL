"""Swimmer Environment Implementation
Action Space
    Box(-1, 1, (2,), float32)
Observation Space
    Box(-inf, inf, (11,), float64)
"""
# Libraries
import gymnasium as gym
import numpy as np
from envs.utils import ActionBoundsIdx
from envs.base_env import MujocoBase


class Reacher(MujocoBase):
    """Reacher Environment"""
    def __init__(
            self, horizon: int = 0, gamma: float = 0.99, verbose: bool = False,
            render: bool = False, clip: bool = True
    ) -> None:
        super().__init__(
            horizon=horizon,
            gamma=gamma,
            verbose=verbose,
            clip=clip
        )
        self.render = render
        render_mode = None
        if self.render:
            render_mode = "human"

        self.gym_env = gym.make(
            'Reacher-v5',
            render_mode=render_mode
        )
        self.action_bounds = [-1, 1]
        self.state_dim = self.gym_env.observation_space.shape[0]    # 11
        self.action_dim = self.gym_env.action_space.shape[0]        # 2
        self.state = None
        self.action_space = self.gym_env.action_space
        self.observation_space = self.gym_env.observation_space
        return

    def step(self, action):
        if self.clip:
            clipped_action = np.clip(
                action,
                self.action_bounds[ActionBoundsIdx.lb],
                self.action_bounds[ActionBoundsIdx.ub],
                dtype=np.float64
            )
        else:
            clipped_action = action
        return super().step(action=clipped_action)
