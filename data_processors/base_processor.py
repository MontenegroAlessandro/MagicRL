"""Base Policy class implementation"""
# Libraries
from abc import ABC, abstractmethod


# Class
class BaseProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def transform(self, state):
        pass