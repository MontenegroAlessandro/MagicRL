"""Base Policy class implementation"""
# Libraries
from abc import ABC, abstractmethod


# Class
class BaseProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dim_feat = None
        
    @abstractmethod
    def transform(self, state):
        pass