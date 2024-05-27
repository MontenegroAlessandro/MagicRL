"""Implementation of the index map between LQR states"""
from data_processors.base_processor import BaseProcessor
import numpy as np
from copy import deepcopy

class LQRTabularProcessor(BaseProcessor):
    def __init__(self, index_map: np.ndarray) -> None:
        super().__init__()
        self.index_map = deepcopy(index_map)

    def transform(self, state: np.ndarray):
        state_idx = int(np.where((self.index_map[:,0] == state[0]) & (self.index_map[:,1] == state[1]))[0].item())
        return state_idx