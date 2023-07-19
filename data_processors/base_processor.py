"""
Summary: Base Policy class implementation
Author: Alessandro Montenegro
Date: 19/7/2023
"""
# Libraries
from abc import ABC, abstractmethod

# Class
class BaseProcessor(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def transform(self, state):
        pass