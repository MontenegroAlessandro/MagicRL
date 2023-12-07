"""
Summary: Base Policy class implementation
Author: Alessandro Montenegro
Date: 19/7/2023
"""
# Libraries
from abc import ABC, abstractmethod


# Class
class BasePolicy(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def draw_action(self, state):
        pass
    
    @abstractmethod
    def set_parameters(self, thetas):
        pass

    @abstractmethod
    def compute_score(self, state, action):
        pass