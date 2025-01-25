"""Inverted Pendulum Environment Implementation
Action Space
    Box(-3, 3, (1,), float32)
Observation Space
    Box(-inf, inf, (4,), float64)
"""
# Libraries
import gymnasium as gym
from envs.base_env import MujocoBase
import numpy as np
from envs.utils import ActionBoundsIdx


class InvertedPendulum(MujocoBase):
    """Inverted Pendulum Wrapper for the environment by GYM."""
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
            'InvertedPendulum-v5',
            render_mode=render_mode
        )
        self.action_bounds = [-3, 3]
        self.state_dim = self.gym_env.observation_space.shape[0]    # 1
        self.action_dim = self.gym_env.action_space.shape[0]        # 4
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
        clipped_action = np.atleast_1d(clipped_action)
        return super().step(action=clipped_action)
    

class CostInvertedPendulum(InvertedPendulum):
    def __init__(
        self, horizon: int = 0, gamma: float = 0.99, verbose: bool = False, 
        forward_reward_weight: float = 1, ctrl_cost_weight: float = 0.1, 
        reset_noise_scale: float = 0.1, 
        exclude_current_positions_from_observation: bool = True, 
        render: bool = False, clip: bool = True
    ) -> None:
        super().__init__(
            horizon=horizon, 
            gamma=gamma, 
            verbose=verbose, 
            forward_reward_weight=forward_reward_weight, 
            ctrl_cost_weight=ctrl_cost_weight, 
            reset_noise_scale=reset_noise_scale, 
            exclude_current_positions_from_observation=exclude_current_positions_from_observation, 
            render=render, 
            clip=clip
        )
        self.with_costs = True
        self.how_many_costs = 1
    
    def step(self, action):
        # compute the cost
        cost = 0
        
        clipped_action = np.clip(
            action,
            self.action_bounds[ActionBoundsIdx.lb],
            self.action_bounds[ActionBoundsIdx.ub],
            dtype=np.float64
        )
        
        slack = action - clipped_action
        if slack.any() > 0:
            cost = np.linalg.norm(slack)
        
        state, rew, done, info = super().step(action)
        
        info["costs"] = np.array([cost], dtype=np.float64)
        
        return state, rew, done, info
