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
            dim_state: int = 1,
            dim_action: int = 1,
            multi_linear: bool = False
    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[LinPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        # Additional attributes
        self.multi_linear = multi_linear
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.tot_params = dim_state * dim_action

        return

    def draw_action(self, state) -> float:
        if len(state) != self.dim_state:
            err_msg = f"[LinPolicy] the state has not the same dimension of the parameter vector:"
            err_msg += f"{len(state)} vs. {self.dim_state}"
            raise ValueError(err_msg)
        action = self.parameters @ state
        return action

    def reduce_exploration(self):
        raise NotImplementedError("[LinPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        if not self.multi_linear:
            self.parameters = copy.deepcopy(thetas)
        else:
            self.parameters = np.array(np.split(thetas, self.dim_action))

    def compute_score(self, state, action) -> np.array:
        if self.multi_linear:
            state = np.tile(state, self.dim_action)
        return state
