"""Base Policy class implementation"""
# Libraries
from abc import ABC, abstractmethod


# Class
class BasePolicy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dim = None
    
    @abstractmethod
    def draw_action(self, state):
        pass
    
    @abstractmethod
    def set_parameters(self, thetas):
        pass

    @abstractmethod
    def compute_score(self, state, action):
        pass

    @abstractmethod
    def reduce_exploration(self):
        pass
