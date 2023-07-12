"""
Summary: Python Script implementing the GridWorldMDP
Author: Alessandro Montenegro
Date: 12/7/2023
"""

# Libraries
import numpy as np
from envs.base_env import BaseEnv

# MACROS
LEGAL_ACTIONS = ["up", "down", "right", "left"]
LEGAL_REWARDS = ["sparse", "linear"]
LEGAL_ENV_CONFIG = ["empty", "walls", "rooms"]

# Class State
class GridWorldState:
    """State for the GridWorld Environment", the position of the agent."""
    def __init__(self) -> None:
        """Summary: Initialization"""
        self.agent_pos = {"x": 0, "y": 0}

# Class Environment       
class GridWorldEnv(BaseEnv):
    """GridWorld Environment"""
    def __init__(
            self, horizon: int = 0, gamma: float = 0, grid_size: int = 0,
            reward_type: str = "linear", env_type: str = "empty"
            ) -> None:
        """
        Summary: Initializaiton function
        Args: 
            horizon (int): look BaseEnv.
            
            gamma (float): look BaseEnv.
            
            grid_size (int): the size of the grid (default to 0).
            
            reward_type (str): how to shape the rewards, legal values in 
            LEGAL_REWARDS (default "linear").

            env_type (str): if and which obstacles to embody in the environment,
            legal values in LEGAL_ENV_CONFIG (default "empty").
        """
        # Super class initialization
        super().__init__(horizon=horizon, gamma=gamma)

        # State initialization
        self.state = GridWorldState()

        # Map initialization
        if grid_size % 2 == 0:
            grid_size += 1
        self.grid_size = grid_size
        self.grid_map = np.zeros((grid_size, grid_size), dtype=int)
        self.grid_view = np.empty((grid_size, grid_size), dtype=str)
        self.grid_view[:, :] = " "
        self.goal_pos = {"x": int(self.grid_size/2), "y": int(self.grid_size/2)}

        # Update the map with the specidfied configuration
        self.forbidden_coordinates = {"x": [], "y": []}
        assert env_type in LEGAL_ENV_CONFIG, "[ERROR] Illegal environment configuration."
        self._load_env_config(env_type=env_type)

        assert reward_type in LEGAL_REWARDS, "[ERROR] Illegal reward type."
        self._load_env_reward(reward_type=reward_type)

        self.grid_view[self.state.agent_pos["x"], self.state.agent_pos["y"]] = "A"

    def step(self, action: str = None):
        """
        Summary: function implementing a step of the environment.
        Args:
            action (str): the action to apply to the environment.
        """
        super().step()
        return
    
    def _load_env_config(self, env_type: str) -> None:
        """
        Summary: this function directly modifies the grid configuration.
        Simple walls or room can be added.

        Args:
            env_type (str): specifies the configuration to use. Accepted 
            values in LEGAL_ENV_CONFIG.
        """
        if env_type == "empty":
            pass
        elif env_type == "walls":
            for i in range(int(self.grid_size / 2)):
                # sample coordinates
                rand_x = np.random.choice(np.arange(self.grid_size))
                rand_y = np.random.choice(np.arange(self.grid_size))
                
                # add forbidden coordinates
                self.forbidden_coordinates["x"].append(rand_x)
                self.forbidden_coordinates["y"].append(rand_y)
                self.grid_view[rand_x, rand_y] = "W"

                # add
                wall_len = np.random.choice(np.arange(int(self.grid_size / 3)))
                coo = np.random.choice(["x", "y"])
                direction = np.random.choice([-1, 1])
                for j in range(wall_len):
                    new_x = rand_x
                    new_y = rand_y
                    new_coo = None
                    if coo == "x":
                        new_x += direction
                        new_coo = new_x
                    else:
                        new_y += direction
                        new_coo = new_y
                    if new_coo < 0 or new_coo >= self.grid_size:
                        break
                    else:
                        self.forbidden_coordinates["x"].append(new_x)
                        self.forbidden_coordinates["y"].append(new_y)
                        self.grid_view[new_x, new_y] = "W"
        elif env_type == "rooms":
            pass
        else:
            pass

    def _load_env_reward(self, reward_type: str) -> None:
        """
        Summary: this function directly defines the reward configuration.
        Linear or sparse reward.

        Args:
            reward_type (str): specifies the configuration to use. Accepted 
            values in LEGAL_REWARDS.
        """
        if reward_type == "linear":
            pass
        elif reward_type == "sparse":
            self.grid_map[:, :] = -1
        else:
            pass
        self.grid_map[self.goal_pos["x"], self.goal_pos["y"]] = 1
        self.grid_view[self.goal_pos["x"], self.goal_pos["y"]] = "G"
