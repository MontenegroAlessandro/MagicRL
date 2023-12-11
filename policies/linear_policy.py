"""
Summary: Implementation of a Linear Policy
Author: @MontenegroAlessandro
Date: 11/12/2023
"""
# Libraries
from policies import BasePolicy
from abc import ABC
import numpy as np
import copy


class LinearPolicy(BasePolicy, ABC):
    """
    Implementation of a Linear Policy in the state vector.
    Thus, the action will be: parameters @ state.
    """
    def __init__(
            self, parameters: np.array = None,
            action_bounds: list = None
    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[GaussPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        err_msg = "[GaussPolicy] too many bounds, 2 or 0 values expected!"
        assert len(action_bounds) == 2 or action_bounds is None, err_msg
        self.action_bounds = action_bounds

        # Additional attributes
        self.dim = len(self.parameters)

        return

    def draw_action(self, state) -> float:
        if len(state) != self.dim:
            err_msg = "[GaussPolicy] the state has not the same dimension of the parameter vector."
            raise ValueError(err_msg)
        action = self.parameters @ state
        if self.action_bounds is not None:
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1], dtype=np.float128)
        return action

    def reduce_exploration(self):
        raise NotImplementedError("[LinearPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        self.parameters = copy.deepcopy(thetas)

    def compute_score(self, state, action) -> np.array:
        return state
