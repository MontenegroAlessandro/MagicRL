"""This policy is sampling uniformly an action to play."""

# Libraries
from policies import BasePolicy
from abc import ABC
from envs import BaseEnv
import numpy as np


class UniformRandomPolicy(BasePolicy, ABC):
    def __init__(
        self,
        env: BaseEnv = None
    ) -> None:
        super().__init__()
        
        assert env is not None, "[UnifPol] No env provided."
        self.env = env
        self.tot_params = 0
        self.dim_action = self.env.action_dim
        self.dim_state = self.env.state_dim
        
    def draw_action(self, state) -> np.array:
        return self.env.action_space.sample()

    def reduce_exploration(self):
        raise NotImplementedError("[UnifPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        raise NotImplementedError("[UnifPolicy] Ops, not implemented yet!")
            
    def get_parameters(self):
        raise NotImplementedError("[UnifPolicy] Ops, not implemented yet!")

    def compute_score(self, state, action) -> np.array:
        raise NotImplementedError("[UnifPolicy] Ops, not implemented yet!")
    
    def diff(self, state):
        raise NotImplementedError("[UnifPolicy] Ops, not implemented yet!")