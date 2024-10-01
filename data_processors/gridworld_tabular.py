"""Implementation of the index map between Gridworld Position"""
from data_processors.base_processor import BaseProcessor
import numpy as np
from copy import deepcopy
from envs.gridworld_env import GridWorldState

class GWTabularProcessor(BaseProcessor):
    def __init__(self, index_map: np.ndarray) -> None:
        super().__init__()
        self.index_map = deepcopy(index_map)

    def transform(self, state: GridWorldState):
        state_idx = self.index_map[state.agent_pos.x, state.agent_pos.y]
        return state_idx