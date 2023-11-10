"""
Smmary: Utils classes and functions for environments
Author: @MontenegroAlessandro
Date: 12/7/2023
"""
# Libraries
import numpy as np
from copy import deepcopy


# Position Class
class Position:
    """Class implementing a position"""

    def __init__(self, x: float, y: float) -> None:
        """
        Args:
            x (int): x-axis coordinate
            y (int): y-axis coordinate
        """
        self.x = x
        self.y = y


# Move for continuous gridworld env
class GWContMove:
    """Class implementing a continuous move for the Continuous Grid World Env"""

    def __init__(self, radius: float, theta: float) -> None:
        """
        Args:
            radius (float): radius to move the agent in polar coordinates.
            Must be bounded in [0, 1].
            
            theta (float): angle in degrees regulating the direction of the 
            agent. Must be bounded in [0, 360].
        """
        self.radius = np.clip(radius, 0, 1)
        self.theta = np.clip(theta, 0, 360)


# Obstacles for continuous gridworld env
LEGAL_OBS_TYPE = ["square", "circle", "sector"]


class Obstacle:
    """Class implementing an obstacle for the Continuous Grid World Env"""

    def __init__(self, type: str, features: dict):
        """
        Args:
            type (str): legal types in LEGAL_OBS_TYPE
            
            features (dict): a dictionary specifying what the obstacles seems
            to be. All the coordinates must be positions.
            If type is "SQUARE" -> 4 coordinates position of the polygon, with 
            keys "p1", "p2", "p3", "p4" as:
                p4---------------p3
                |                |
                |                |
                p1---------------p2
            If type is "CIRCLE" -> "radius" and "center" coordinates
        """
        assert type in LEGAL_OBS_TYPE, "[ERROR] Illegal Ostacle type."
        self.type = type
        self.features = deepcopy(features)

        if self.type == "square":
            assert features["p1"].x == features["p4"].x
            assert features["p2"].x == features["p3"].x
            assert features["p1"].y == features["p2"].y
            assert features["p3"].y == features["p4"].y

    def is_in(self, pos: Position) -> bool:
        """
        Summary:
            Function saying if "pos" is in the obstacle

        Args:
            pos (Position): position to verify 

        Returns:
            bool: flag saying if "pos" is inside the obstacle 
        """
        res = False
        if self.type == "square":
            if (self.features["p1"].x <= pos.x <= self.features["p2"].x) and (self.features["p1"].y <= pos.y <= self.features["p4"].y):
                res = True
        elif self.type == "circle":
            dist = (pos.x - self.features["center"].x) ** 2
            dist += (pos.y - self.features["center"].y) ** 2
            dist = np.sqrt(dist)
            if dist <= self.features["radius"]:
                res = True
        else:
            pass
        return res


def design_u_obstacle(grid_size: int):
    edge1 = Obstacle(
        type="square",
        features={"p1": Position(grid_size / 2 - 1, grid_size / 2 - 1),
                  "p2": Position(grid_size / 2 + 1, grid_size / 2 - 1),
                  "p3": Position(grid_size / 2 + 1, grid_size / 2 - .5),
                  "p4": Position(grid_size / 2 - 1, grid_size / 2 - .5)}
    )

    edge2 = Obstacle(
        type="square",
        features={"p1": Position(grid_size / 2 - 1, grid_size / 2 - 1),
                  "p2": Position(grid_size / 2 - .5, grid_size / 2 - 1),
                  "p3": Position(grid_size / 2 - .5, grid_size / 2 + 1),
                  "p4": Position(grid_size / 2 - 1, grid_size / 2 + 1)}
    )

    edge3 = Obstacle(
        type="square",
        features={"p1": Position(grid_size / 2 + .5, grid_size / 2 - 1),
                  "p2": Position(grid_size / 2 + 1, grid_size / 2 - 1),
                  "p3": Position(grid_size / 2 + 1, grid_size / 2 + 1),
                  "p4": Position(grid_size / 2 + .5, grid_size / 2 + 1)}
    )
    return [edge1, edge2, edge3]
