""" 
Summary: Data Processing class performing a data transformation identity
Author: @MontenegroAlessandro
Date: 20/7/2023
"""
# Libraries
from data_processors.base_processor import BaseProcessor

# Class
class IdentityDataProcessor(BaseProcessor):
    """Identity Data Processor, used for default values"""
    def __init__(self) -> None:
        super().__init__()
    
    def transform(self, state):
        return state