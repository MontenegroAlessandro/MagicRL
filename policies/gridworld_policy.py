"""
Summary: implementation of a parametric policy for the continuous verison of 
Grid World.
Author: @MontenegroAlessandro
Date: 19/7/2023
"""
# Libraries
from policies.base_policy import BasePolicy
import numpy as np

# Action class
class GWContAction:
    def __init__(self, radius: float = 0, theta: float = 0) -> None:
        """
        Args:
            radius (float, optional): radius of the action. Defaults to 0.
            theta (float, optional): orientation of the action. Defaults to 0.
        """
        self.radius = radius
        self.theta = theta

# Policy Implementation
class GWPolicy(BasePolicy):
    """
    Grid World parametric policy for GridWorldEnvCont.
    The policy controls both the radius and the angle of the next move.
    """
    def __init__(self, thetas: list, dim_state: int) -> None:
        """
        Args:
            thetas (list): Parameter initialization for the policy "[, Ω]"
            dim_state (int): size of the state (features)
        """
        super().__init__()
        self.dim_state = dim_state
        self.thetas = np.array(thetas[:dim_state])
        self.omegas = np.array(thetas[dim_state:])
    
    def draw_action(self, state: list):
        """
        Summary:
            Basing on the current parameter configuration, returns the next 
            action to compute.
        Args:
            state (list): list of numbers resuming the state of the MDP.

        Returns:
            float, float: radius of the next move, angle of the next move
        """
        state = np.array(state)
        #radius = np.clip(np.exp(self.omegas.T @ state), 0, 1)
        radius = 1 / (1 + np.exp(-self.omegas.T @ state))
        theta = np.rad2deg(np.pi * np.tanh(self.thetas.T @ state)) #+ 180
        return GWContAction(radius=radius, theta=theta)
    
    def set_parameters(self, thetas: list):
        """
        Summary:
            Update the parameter vector
        
        Args:
            thetas (list): new parameters for the policy
        """
        self.thetas = np.array(thetas[:self.dim_state])
        self.omegas = np.array(thetas[self.dim_state:])
