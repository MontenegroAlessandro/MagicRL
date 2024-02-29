"""Half Cheetah Environment Implementation
Action Space
    Box(-1, 1, (6,), float32)
Observation Space
    Box(-inf, inf, (17,), float64)
"""
# Libraries
import gymnasium as gym
from envs.base_env import MujocoBase
import numpy as np
from envs.utils import ActionBoundsIdx


class HalfCheetah(MujocoBase):
    """Half Cheetah Wrapper for the environment by GYM."""
    def __init__(
            self, horizon: int = 0, gamma: float = 0.99, verbose: bool = False,
            forward_reward_weight: float = 1.0,
            ctrl_cost_weight: float = 0.1,
            reset_noise_scale: float = 0.1,
            exclude_current_positions_from_observation: bool = True,
            render: bool = False,
            clip: bool = True
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
