"""
Smmary: Utils classes and functions for environments
Author: Alessandro Montenegro
Date: 12/7/2023
"""
class Position:
    """Class implementing a position"""
    def __init__(self, x: int, y: int) -> None:
        """
        Args:
            x (int): x axis coordinate
            y (int): y axis coordinate
        """
        self.x = x
        self.y = y