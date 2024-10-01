"""Base Policy class implementation"""
# Libraries
from abc import ABC, abstractmethod


# Class
class BasePolicy(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dim_state = None
        self.dim_action = None
        self.tot_params = None
    
    @abstractmethod
    def draw_action(self, state):
        pass
    
    @abstractmethod
    def set_parameters(self, thetas, *args, **kwargs):
        pass

    @abstractmethod
    def compute_score(self, state, action):
        pass
    
    @abstractmethod
    def diff(self, state):
        pass

    @abstractmethod
    def reduce_exploration(self):
        pass
    
    @abstractmethod
    def get_parameters(self):
        pass
