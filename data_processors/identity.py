""" Data Processing class performing a data transformation identity"""
# Libraries
from data_processors.base_processor import BaseProcessor


# Class
class IdentityDataProcessor(BaseProcessor):
    """Identity Data Processor, used for default values"""
    def __init__(self, dim_feat=None) -> None:
        super().__init__()
        self.dim_feat = dim_feat
    
    def transform(self, state):
        return state
