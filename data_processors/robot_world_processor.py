"""
Data Processing class performing a data transformation of the state
using the following feature map: s = [1, \sqrt(|s|, s, s^2)]
"""

# Libraries
import numpy as np

from data_processors import BaseProcessor
from data_processors.base_processor import BaseProcessor

# Data Processor class
class RobotWorldProcessor(BaseProcessor):
    """
    Data Processor Class Mapping a state of RobotWorld Continuous environment
    into a feature vector.
    """
    def __init__(self) -> None:
        """
        Args:
            num_basis (int): how many Gaussians to use
            grid_size (int): the dimension of the gridworld
            std_dev (float): the standard deviation that each gaussian needs
            to have
        """
        super().__init__()

    def transform(self, state: np.array) -> np.array:
        """
        Summary:
            Compute the mapping from the state to the feature vector using
            the gaussians

        Args:
            state (np.array): the current state of the agent in the Robot
            World Continuous Env

        Returns:
            np.array: feature mapping vector
        """
        # Ensure each component is a NumPy array and concatenate them
        ones = np.array([1])  # Make `1` into a 1-element array
        sqrt_abs_state = np.sqrt(np.abs(state))
        state_squared = np.clip(state ** 2, 0, 100)

        # Concatenate all components to create a single homogeneous array
        return np.concatenate([ones, sqrt_abs_state, state, state_squared])