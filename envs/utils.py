"""Utils classes and functions for environments"""
# Libraries
import numpy as np
from copy import deepcopy


class ActionBoundsIdx:
    lb = 0
    ub = 1


class StateBoundsIdx:
    lb = 0
    ub = 1


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


def design_u_obstacle(grid_size: int, size: float = 0.1) -> list:
    # Define the barycenter of the U-shape
    center_x = grid_size / 2
    center_y = grid_size / 2

    # Set dimensions for the U-shape
    horizontal_width = 3  # Width of the horizontal bottom edge
    vertical_height = 2   # Height of each vertical edge

    # Bottom horizontal edge of the U-shape, centered horizontally on the barycenter
    edge1 = Obstacle(
        type="square",
        features={
            "p1": Position(center_x - horizontal_width / 2, center_y - vertical_height / 2 - size),
            "p2": Position(center_x + horizontal_width / 2, center_y - vertical_height / 2 - size),
            "p3": Position(center_x + horizontal_width / 2, center_y - vertical_height / 2),
            "p4": Position(center_x - horizontal_width / 2, center_y - vertical_height / 2)
        }
    )

    # Left vertical edge of the U-shape, positioned to the left of the barycenter
    edge2 = Obstacle(
        type="square",
        features={
            "p1": Position(center_x - horizontal_width / 2, center_y - vertical_height / 2),
            "p2": Position(center_x - horizontal_width / 2 + size, center_y - vertical_height / 2),
            "p3": Position(center_x - horizontal_width / 2 + size, center_y + vertical_height / 2),
            "p4": Position(center_x - horizontal_width / 2, center_y + vertical_height / 2)
        }
    )

    # Right vertical edge of the U-shape, positioned to the right of the barycenter
    edge3 = Obstacle(
        type="square",
        features={
            "p1": Position(center_x + horizontal_width / 2 - size, center_y - vertical_height / 2),
            "p2": Position(center_x + horizontal_width / 2, center_y - vertical_height / 2),
            "p3": Position(center_x + horizontal_width / 2, center_y + vertical_height / 2),
            "p4": Position(center_x + horizontal_width / 2 - size, center_y + vertical_height / 2)
        }
    )

    return [edge1, edge2, edge3]
