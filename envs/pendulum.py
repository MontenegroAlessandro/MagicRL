import gymnasium as gym
import numpy as np
from envs.utils import ActionBoundsIdx
from envs.base_env import MujocoBase


class Pendulum(MujocoBase):
    """Pendulum Wrapper for the environment by GYM."""
    def __init__(self, horizon: int = 0, gamma: float = 0.99, verbose: bool = False,
            render: bool = False, clip: bool = True) -> None:
        super().__init__(horizon=horizon, gamma=gamma, verbose=verbose, clip=clip)
        self.render = render
        render_mode = None
        if self.render:
            render_mode = "human"
            
        self.gym_env = gym.make(
            "InvertedPendulum-v5",
            render_mode=render_mode
        )
        self.action_bounds = [-3, 3]
        self.state_dim = self.gym_env.observation_space.shape[0]    # 4
        self.action_dim = self.gym_env.action_space.shape[0]        # 1
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