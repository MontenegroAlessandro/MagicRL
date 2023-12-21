"""Implementation of a Linear Policy"""
# Libraries
from policies import BasePolicy
from abc import ABC
import numpy as np
import copy


class NeuralNetworkPolicy(BasePolicy, ABC):
    def __init__(self):
        super().__init__()

    def draw_action(self, state) -> float:
        pass

    def reduce_exploration(self):
        raise NotImplementedError("[NNPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        pass

    def compute_score(self, state, action) -> np.array:
        return state
