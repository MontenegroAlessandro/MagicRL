"""
Summary: Implementation of a Gaussian Policy
Author: @MontenegroAlessandro
Date: 7/12/2023
"""
# Libraries
from policies import BasePolicy
from abc import ABC
import numpy as np
import copy


class LinearGaussianPolicy(BasePolicy, ABC):
    """
    Implementation of a Gaussian Policy which is linear in the state.
    Thus, the mean will be: parameters @ state.
    The standard deviation is fixed and is defined by the user.
    """
    def __init__(
            self, parameters: np.array = None,
            std_dev: float = 0.1,
            action_bounds: list = None

    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[GaussPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        err_msg = "[GaussPolicy] standard deviation is negative!"
        assert std_dev > 0, err_msg
        self.std_dev = std_dev

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
        mean = self.parameters @ state
        action = np.random.normal(mean, self.std_dev)
        if self.action_bounds is not None:
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1], dtype=float)
        return action

    def set_parameters(self, thetas) -> None:
        self.parameters = copy.deepcopy(thetas)

    def compute_score(self, state, action) -> np.array:
        scores = np.zeros(self.dim, dtype=float)
        scores[:] = (action - self.parameters @ state) / (self.std_dev ** 2)
        return scores
