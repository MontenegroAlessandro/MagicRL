"""
Summary: Python Script implementing the GridWorldMDP with goal in the center
Author: @MontenegroAlessandro
Date: 12/7/2023
"""

# Libraries
import numpy as np
from envs.base_env import BaseEnv
from envs.utils import Position, GWContMove, Obstacle
from copy import deepcopy
import json, os, io, errno
import matplotlib.pyplot as plt
import glob
from PIL import Image

# MACROS
LEGAL_ACTIONS = ["up", "down", "right", "left"]
LEGAL_REWARDS = ["sparse", "linear"]
LEGAL_ENV_CONFIG = ["empty", "walls", "rooms"]


# Class State
class GridWorldState:
    """State for the GridWorld Environment", the position of the agent."""

    def __init__(self, x: int = 0, y: int = 0) -> None:
        """Summary: Initialization
        Args:
            x (int): x axis coordinate of the agent (default 0)
            y (int): y axis coordinate of the agent (default 0)
        """
        self.agent_pos = Position(x=x, y=y)


# Class Environment
class GridWorldEnvDisc(BaseEnv):
    """GridWorld Environment Discrete"""

    def __init__(
            self, horizon: int = 0, gamma: float = 0, grid_size: int = 0,
            reward_type: str = "linear", env_type: str = "empty",
            render: bool = False, dir: str = None
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
            
            render (bool): flag indicating whether to save graphically the results.
            
            dir (str): directory in which save the results.
        """
        # Super class initialization
        super().__init__(horizon=horizon,
                         gamma=gamma)  # self.horizon, self.gamma, self.time

        # State initialization
        self.state = GridWorldState()

        # Map initialization
        if grid_size % 2 == 0:
            grid_size += 1
        self.grid_size = grid_size
        self.grid_map = np.zeros((grid_size, grid_size), dtype=int)
        # self.grid_view = np.empty((grid_size, grid_size), dtype=str)
        # self.grid_view[:, :] = " "
        self.goal_pos = Position(x=int(self.grid_size / 2),
                                 y=int(self.grid_size / 2))

        # Update the map with the specidfied configuration
        self.forbidden_coordinates = []
        assert env_type in LEGAL_ENV_CONFIG, "[ERROR] Illegal environment configuration."
        self._load_env_config(env_type=env_type)

        assert reward_type in LEGAL_REWARDS, "[ERROR] Illegal reward type."
        self._load_env_reward(reward_type=reward_type)

        # self.grid_view[self.state.agent_pos.x, self.state.agent_pos.y] = "A"

        # Saving parameters
        self.render = render
        self.dir = dir
        self.episode_conuter = 1
        if self.render and self.dir is None:
            print("[Error] No directory provided.")
            quit(-1)
        self.history = {"0": {
            "pos_x": int(self.state.agent_pos.x),
            "pos_y": int(self.state.agent_pos.y),
            "r": int(
                self.grid_map[self.state.agent_pos.y, self.state.agent_pos.x]),
            "abs": int(False)
        }}
        if self.render:
            self.save_single_image()

    def step(self, action: str = None):
        """
        Summary: function implementing a step of the environment.
        Args:
            action (str): the action to apply to the environment.
        Returns:
            new position and reward
        """
        # Update of self.time
        self.time += 1

        # Check the action correctness
        assert action in LEGAL_ACTIONS, "[ERROR] Illegal action taken."

        # Compute the next state
        new_pos = deepcopy(self.state.agent_pos)
        if action == "up":
            new_pos.y = (new_pos.y + 1) % self.grid_size
        elif action == "down":
            new_pos.y = (new_pos.y - 1) % self.grid_size
        elif action == "right":
            new_pos.x = (new_pos.x + 1) % self.grid_size
        elif action == "left":
            new_pos.x = (new_pos.x - 1) % self.grid_size
        else:
            pass

        # Check if the new position is forbidden
        isAbs = self.is_absorbing(self.state.agent_pos)
        if new_pos not in self.forbidden_coordinates and not isAbs:
            # Update map
            # self.grid_view[self.state.agent_pos.y, self.state.agent_pos.x] = " "
            # self.grid_view[new_pos.y, new_pos.x] = "A"

            # Update state
            self.state.agent_pos = deepcopy(new_pos)
        else:
            new_pos = deepcopy(self.state.agent_pos)

        # Save results
        reward = self.grid_map[new_pos.y, new_pos.x]
        isNewAbs = self.is_absorbing(position=new_pos)
        self.history[str(self.time)] = {
            "pos_x": int(new_pos.x),
            "pos_y": int(new_pos.y),
            "r": int(reward),
            "abs": int(isNewAbs)
        }
        if self.render:
            self.save_single_image()
        return new_pos, reward, isNewAbs

    def reset(self) -> None:
        # self.time init
        super().reset()

        # State initialization
        self.state = GridWorldState()

        # Map initialization
        # self.grid_view = np.empty((self.grid_size, self.grid_size), dtype=str)
        # self.grid_view[:, :] = " "
        # self.grid_view[self.state.agent_pos.x, self.state.agent_pos.y] = "A"

        # Save results
        if self.dir is not None:
            self.save_history()
            if self.render:
                self.save_gif()
        self.episode_conuter += 1

        # History
        self.history = {"0": {
            "pos_x": int(self.state.agent_pos.x),
            "pos_y": int(self.state.agent_pos.y),
            "r": int(
                self.grid_map[self.state.agent_pos.y, self.state.agent_pos.x]),
            "abs": int(False)
        }}
        if self.render:
            self.save_single_image()

    def sample_random_state(self):
        pass

    def save_history(self) -> None:
        """Save json of the history"""
        name = self.dir + f"/GridWorldResults_{self.episode_conuter}.json"
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.history, ensure_ascii=False, indent=4))
            f.close()

    def save_single_image(self) -> None:
        """Save an image of the current state of the environment"""
        # clear plot
        plt.clf()
        plt.xlim(0, self.grid_size - 1)
        plt.ylim(0, self.grid_size - 1)
        plt.xticks(np.arange(self.grid_size))
        plt.yticks(np.arange(self.grid_size))

        # goal pos
        plt.plot(self.goal_pos.x, self.goal_pos.y, "o", color="green",
                 label="GOAL")

        # agent
        plt.plot(self.state.agent_pos.x, self.state.agent_pos.y, "o",
                 color="blue", label="AGENT")

        # plot walls
        for elem in self.forbidden_coordinates:
            plt.plot(elem.x, elem.y, "o", color="red")

        # plot positions
        plt.grid()
        plt.legend(loc="best")
        plt.title(f"GridWorld EPISODE{self.episode_conuter} FRAME {self.time}")

        if self.time < 10:
            name = self.dir + f"/frames_{self.episode_conuter}" + f"/GridWorldFrame_{self.episode_conuter}_0{self.time}.jpeg"
        else:
            name = self.dir + f"/frames_{self.episode_conuter}" + f"/GridWorldFrame_{self.episode_conuter}_{self.time}.jpeg"
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        plt.savefig(name, format="jpeg")

    def save_gif(self) -> None:
        name = self.dir + f"/frames_{self.episode_conuter}"
        frames = [Image.open(image) for image in
                  sorted(glob.glob(f"{name}/*.jpeg"))]
        frame_one = frames[0]
        frame_one.save(f"{self.dir}/GridWorldGif_{self.episode_conuter}.gif",
                       format="GIF", append_images=frames, save_all=True,
                       duration=self.horizon * 100, loop=0)

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
            num_walls = np.random.choice(np.arange(self.grid_size))
            for i in range(num_walls):
                # sample coordinates
                rand_x = np.random.choice(np.arange(self.grid_size))
                rand_y = np.random.choice(np.arange(self.grid_size))

                # add forbidden coordinates
                self.forbidden_coordinates.append(Position(x=rand_x, y=rand_y))
                # self.grid_view[rand_x, rand_y] = "W"

                # add
                wall_len = np.random.choice(np.arange(int(self.grid_size)))
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
                        self.forbidden_coordinates.append(
                            Position(x=new_x, y=new_y))
                        # self.grid_view[new_x, new_y] = "W"
        elif env_type == "rooms":
            # TODO: implement
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
            current_distance = 1
            while self.goal_pos.x - current_distance >= 0 or self.goal_pos.x + current_distance < self.grid_size:
                self.grid_map[self.goal_pos.x + current_distance,
                :] = -current_distance
                self.grid_map[self.goal_pos.x - current_distance,
                :] = -current_distance
                self.grid_map[:,
                self.goal_pos.y + current_distance] = -current_distance
                self.grid_map[:,
                self.goal_pos.y - current_distance] = -current_distance
                current_distance += 1

        elif reward_type == "sparse":
            self.grid_map[:, :] = -1
        else:
            pass
        self.grid_map[self.goal_pos.x, self.goal_pos.y] = 1
        # self.grid_view[self.goal_pos.x, self.goal_pos.y] = "G"

    def is_absorbing(self, position: Position) -> bool:
        """
        Args:
            position (Position): the position to check whether absorbing

        Returns:
            bool: True if "position" is an absorbing state, False otherwise
        """
        return position.x == self.goal_pos.x and position.y == self.goal_pos.y


class GridWorldEnvCont(BaseEnv):
    """GridWorld Environment Continuous"""

    def __init__(
            self, horizon: int = 0, gamma: float = 0, grid_size: int = 0,
            reward_type: str = "linear",
            render: bool = False, dir: str = None, init_state: list = None,
            obstacles: list = None, verbose: bool = False, pacman: bool = False,
            goal_tol: float = 0, obstacles_strict_flag: bool = False
    ) -> None:
        """
        Summary: Initialization function
        Args: 
            horizon (int): look BaseEnv.
            
            gamma (float): look BaseEnv.
            
            grid_size (int): the size of the grid (default to 0).
            
            reward_type (str): how to shape the rewards, legal values in 
            LEGAL_REWARDS (default "linear").

            env_type (str): if and which obstacles to embody in the environment,
            legal values in LEGAL_ENV_CONFIG (default "empty").
            
            render (bool): flag indicating whether to save graphically the results.
            
            dir (str): directory in which save the results.
            
            init_state (list): a list of 2 coordinates stating how to initialize
            the state. If not passed, the state will be initialized randomly.
            Defaults to None.
            
            obstacles (list): list of Obstacles to put in the environment. 
            Defaults to empty list [].
            
            verbose (bool): tells whether to print more information.
            
            pacman (bool): tells whether to employ the pacman effect in the env.
            
            goal_tol (float): degree of tolerance to be in the goal position.
            The absorbing state will be considered if the distance between the
            goal and the agent is epsilon.

            obstacles_strict_flag (bool): a flag telling how to deal with
            obstacles. If its value is "True", then the environment will not
            allow to enter the zones of the obstacles. Elsewhere, if its value
            is "False", then just a negative reward will be assigned when the
            agent is in a forbidden zone. Default to "False".
        """
        # Super class initialization
        super().__init__(horizon=horizon, gamma=gamma,
                         verbose=verbose)  # self.horizon, self.gamma, self.time

        # Map initialization
        if obstacles is None:
            obstacles = []
        self.grid_size = grid_size
        self.goal_pos = Position(x=self.grid_size / 2,
                                 y=self.grid_size / 2)
        self.pacman = pacman
        self.epsilon = goal_tol

        # State initialization
        self.init_state = init_state
        if init_state is not None:
            self.state = GridWorldState(x=init_state[0], y=init_state[1])
        else:
            self.state = self.sample_random_state()

        # Reward
        assert reward_type in LEGAL_REWARDS, "[ERROR] Illegal reward type."
        self.reward_type = reward_type

        # Obstacles
        self.obstacles = obstacles
        self.obstacles_strict_flag = obstacles_strict_flag

        # Saving parameters
        self.render = render
        self.dir = dir
        self.episode_counter = 1
        if self.render and self.dir is None:
            print("[Error] No directory provided.")
            quit(-1)
        self.history = {"0": {
            "pos_x": self.state.agent_pos.x,
            "pos_y": self.state.agent_pos.y,
            "r": self.reward(),
            "abs": int(False)
        }}
        if self.render:
            self.save_single_image()

    def sample_random_state(self):
        # FIXME: remove the int()
        space = np.linspace(start=0, stop=self.grid_size, num=100)
        return GridWorldState(x=int(np.random.choice(space)),
                              y=int(np.random.choice(space)))

    def step(self, action: GWContMove = None):
        """
        Summary: function implementing a step of the environment.
        Args:
            action (GWContMove): the action to apply to the environment.
        Returns:
            new position, reward and absorbing flag.
        """
        # Update of self.time
        self.time += 1
        outside_flag = False

        # Compute the next state
        new_pos = deepcopy(self.state.agent_pos)
        new_pos.x = action.radius * np.cos(
            np.deg2rad(action.theta)) + self.state.agent_pos.x
        new_pos.y = action.radius * np.sin(
            np.deg2rad(action.theta)) + self.state.agent_pos.y
        if self.pacman:
            new_pos.x = new_pos.x % self.grid_size
            new_pos.y = new_pos.y % self.grid_size
        else:
            clipped_x = np.clip(new_pos.x, 0, self.grid_size)
            clipped_y = np.clip(new_pos.y, 0, self.grid_size)
            if clipped_x != new_pos.x or clipped_y != new_pos.y:
                outside_flag = True
                new_pos.x = clipped_x
                new_pos.y = clipped_y

        # Check if the new position is forbidden
        isAbs = self.is_absorbing(self.state.agent_pos)
        forbidden = False
        for obs in self.obstacles:
            if obs.is_in(pos=new_pos):
                forbidden = True
                break
        if (not forbidden and not isAbs) or \
                (forbidden and not self.obstacles_strict_flag):
            # Update state
            self.state.agent_pos = deepcopy(new_pos)
        else:
            # Reload old state
            new_pos = deepcopy(self.state.agent_pos)

        # Save results
        reward = self.reward(outside=outside_flag, forbidden=forbidden)
        isNewAbs = self.is_absorbing(position=new_pos)
        self.history[str(self.time)] = {
            "pos_x": int(new_pos.x),
            "pos_y": int(new_pos.y),
            "r": int(reward),
            "abs": int(isNewAbs)
        }
        if self.render:
            self.save_single_image()

        if isNewAbs:
            return new_pos, (self.horizon - self.time) * reward, True
        else:
            return new_pos, reward, isNewAbs

    def reset(self) -> None:
        # self.time init
        super().reset()

        # State initialization
        if self.init_state is not None:
            self.state = GridWorldState(x=self.init_state[0],
                                        y=self.init_state[1])
        else:
            self.state = self.sample_random_state()

        # Save results
        if self.dir is not None:
            self.save_history()
            if self.render:
                self.save_gif()
        self.episode_counter += 1

        # History
        self.history = {"0": {
            "pos_x": self.state.agent_pos.x,
            "pos_y": self.state.agent_pos.y,
            "r": self.reward(),
            "abs": int(False)
        }}
        if self.render:
            self.save_single_image()

    def reward(self, outside=False, forbidden=False):
        if forbidden:
            # If the agent is in a forbidden zone, punish it with a super
            # negative reward
            return -(self.grid_size ** 2)
        else:
            # Distance computation
            dist = (self.state.agent_pos.x - self.goal_pos.x) ** 2
            dist += (self.state.agent_pos.y - self.goal_pos.y) ** 2
            dist = np.sqrt(dist)

            # Penalization if the policy said to go outside
            dist += 10 if outside else 0

            # Give reward
            if dist <= self.epsilon:
                return 1
            if self.reward_type == "linear":
                return -dist
            elif self.reward_type == "sparse":
                return -1
            else:
                raise NotImplementedError

    def is_absorbing(self, position: Position):
        dist = np.sqrt((self.goal_pos.x - position.x) ** 2 + (
                    self.goal_pos.y - position.y) ** 2)
        # return position.x == self.goal_pos.x and position.y == self.goal_pos.y
        return dist <= self.epsilon

    def save_history(self) -> None:
        """Save json of the history"""
        name = self.dir + f"/GridWorldContResults_{self.episode_counter}.json"
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(self.history, ensure_ascii=False, indent=4))
            f.close()

    def save_single_image(self) -> None:
        """Save an image of the current state of the environment"""
        # clear plot
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(self.grid_size + 1))
        ax.set_yticks(np.arange(self.grid_size + 1))

        # goal pos
        ax.plot(self.goal_pos.x, self.goal_pos.y, "o", color="green",
                label="GOAL")

        # agent
        ax.plot(self.state.agent_pos.x, self.state.agent_pos.y, "o",
                color="blue", label="AGENT")

        # plot walls
        for obs in self.obstacles:
            if obs.type == "square":
                x = obs.features["p1"].x
                y = obs.features["p1"].y
                w = obs.features["p2"].x - obs.features["p1"].x
                h = obs.features["p4"].y - obs.features["p1"].y
                ax.add_patch(plt.Rectangle((x, y), w, h, alpha=0.3, color="red",
                                           label="obstacle"))
            elif obs.type == "circle":
                x = obs.features["center"].x
                y = obs.features["center"].y
                r = obs.features["radius"]
                ax.add_patch(plt.Circle((x, y), r, alpha=0.3, color="orange",
                                        label="obstacle"))

        # plot positions
        plt.grid()
        plt.legend(loc="best")
        plt.title(f"GridWorld EPISODE{self.episode_counter} FRAME {self.time}")

        if self.time < 10:
            name = self.dir + f"/frames_{self.episode_counter}" + f"/GridWorldFrame_{self.episode_counter}_0{self.time}.jpeg"
        else:
            name = self.dir + f"/frames_{self.episode_counter}" + f"/GridWorldFrame_{self.episode_counter}_{self.time}.jpeg"
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        plt.savefig(name, format="jpeg")
        plt.close(fig)

    def save_gif(self) -> None:
        name = self.dir + f"/frames_{self.episode_counter}"
        frames = [Image.open(image) for image in
                  sorted(glob.glob(f"{name}/*.jpeg"))]
        frame_one = frames[0]
        frame_one.save(
            f"{self.dir}/GridWorldContGif_{self.episode_counter}.gif",
            format="GIF", append_images=frames, save_all=True, duration=150,
            loop=0)
