"""Implementation of a Linear Policy"""
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
            action_bounds: list = None,
            multi_linear: bool = False
    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[LinPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        err_msg = "[LinPolicy] too many bounds, 2 or 0 values expected!"
        assert len(action_bounds) == 2 or action_bounds is None, err_msg
        self.action_bounds = action_bounds

        # Additional attributes
        self.multi_linear = multi_linear
        self.dim_action = 1
        if not self.multi_linear:
            self.dim_state = len(self.parameters)
        else:
            self.dim_state = len(self.parameters[0])
            self.dim_action = len(self.parameters)

        return

    def draw_action(self, state) -> float:
        if len(state) != self.dim_state:
            err_msg = f"[LinPolicy] the state has not the same dimension of the parameter vector:"
            err_msg += f"{len(state)} vs {self.dim_state}"
            raise ValueError(err_msg)
        action = self.parameters @ state
        if self.action_bounds is not None:
            action = np.clip(action, self.action_bounds[0], self.action_bounds[1], dtype=np.float128)
        return action

    def reduce_exploration(self):
        raise NotImplementedError("[LinPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        if not self.multi_linear:
            self.parameters = copy.deepcopy(thetas)
        else:
            self.parameters = np.array(np.split(thetas, self.dim_action))

    def compute_score(self, state, action) -> np.array:
        return state
