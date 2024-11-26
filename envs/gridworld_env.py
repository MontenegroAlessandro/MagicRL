"""Python Script implementing the GridWorldMDP with goal in the center"""
import copy

# todo -> make the discrete environment in line with the continuous one
# Libraries
import numpy as np
from envs.base_env import BaseEnv
from envs.utils import Position, GWContMove, Obstacle, design_u_obstacle
from copy import deepcopy
import json, os, io, errno
import matplotlib.pyplot as plt
import glob
from PIL import Image

# MACROS
LEGAL_ACTIONS = ["up", "down", "right", "left"]
ACTIONS = [0, 1, 2, 3]
LEGAL_REWARDS = ["sparse", "linear"]
LEGAL_ENV_CONFIG = ["empty", "walls", "rooms", "U"]


# Class State
class GridWorldState:
    """State for the GridWorld Environment", the position of the agent."""

    def __init__(self, x: float = 0, y: float = 0) -> None:
        """Summary: Initialization
        Args:
            x (int): x-axis coordinate of the agent (default 0)
            y (int): y-axis coordinate of the agent (default 0)
        """
        self.x = x
        self.y = y
        self.agent_pos = Position(x=x, y=y)


# Class Environment
class GridWorldEnvDisc(BaseEnv):
    """GridWorld Environment Discrete"""

    def __init__(
            self, horizon: int = 0, gamma: float = 0, grid_size: int = 0,
            reward_type: str = "linear", env_type: str = "empty",
            render: bool = False, dir: str = None, random_init: bool = False,
            ding_flag: bool = False
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
        """
        # Super class initialization
        super().__init__(horizon=horizon, gamma=gamma)  # self.horizon, self.gamma, self.time
        self.continuous_env = False
        self.ding_flag = ding_flag

        # Map initialization
        if grid_size % 2 == 0:
            grid_size += 1
        self.grid_size = grid_size
        self.action_dim = 4
        self.state_dim = grid_size ** 2
        # State initialization
        self.random_init = random_init
        if self.random_init:
            x = np.random.choice([0, self.grid_size-1])
            y = np.random.choice([0, self.grid_size-1])
            self.state = GridWorldState(x=x, y=y)
        else:
            self.state = GridWorldState()
        self.grid_map = np.zeros((grid_size, grid_size), dtype=int)
        self.discrete_state_space = np.arange(grid_size * grid_size)
        self.discrete_action_space = np.arange(4)
        self.grid_index = self.discrete_state_space.reshape((grid_size, grid_size))
        self.goal_pos = Position(x=int(self.grid_size / 2), y=int(self.grid_size / 2))

        # Update the map with the specified configuration
        self.forbidden_coordinates = []
        assert env_type in LEGAL_ENV_CONFIG, "[ERROR] Illegal environment configuration."
        self._load_env_config(env_type=env_type)

        assert reward_type in LEGAL_REWARDS, "[ERROR] Illegal reward type."
        self._load_env_reward(reward_type=reward_type)

        # costs stuff
        if env_type != "empty":
            self.with_costs = True
            self.how_many_costs = 1

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
            "r": int(self.grid_map[self.state.agent_pos.y, self.state.agent_pos.x]),
            "abs": int(False)
        }}
        if self.render:
            self.save_single_image()

    def step(self, action: int = None):
        """
        Summary: function implementing a step of the environment.
        Args:
            action (int): the action to apply to the environment.
        Returns:
            new position and reward
        """
        # Update of self.time
        self.time += 1

        # Check the action correctness
        assert action in ACTIONS, "[ERROR] Illegal action taken."

        # Compute the next state
        new_pos = deepcopy(self.state.agent_pos)
        if action == 0:
            # UP
            new_pos.y = (new_pos.y + 1) % self.grid_size
        elif action == 1:
            # DOWN
            new_pos.y = (new_pos.y - 1) % self.grid_size
        elif action == 2:
            # RIGHT
            new_pos.x = (new_pos.x + 1) % self.grid_size
        elif action == 3:
            # LEFT
            new_pos.x = (new_pos.x - 1) % self.grid_size
        else:
            raise ValueError(f"[DiscGW] {action} is illegal!")

        # Check if the new position is forbidden
        isAbs = self.is_absorbing(self.state.agent_pos)
        cost = 1 if self.ding_flag else 0
        if self.with_costs:
            if isAbs:
                new_pos = deepcopy(self.state.agent_pos)
            else:
                for pos in self.forbidden_coordinates:
                    if pos.x == new_pos.x and pos.y == new_pos.y:
                        cost = 0 if self.ding_flag else 1
                        break
                self.state.agent_pos = deepcopy(new_pos)

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

        # save cost
        info = dict(costs=deepcopy(np.array([cost], dtype=np.float64)))

        # we can access to the index of the position with:
        # env.state_index[new_pos.x, new_pos.y]
        return new_pos, reward, isNewAbs, info

    def reset(self) -> None:
        # self.time init
        super().reset()

        if self.random_init:
            x = np.random.choice([0, self.grid_size - 1])
            y = np.random.choice([0, self.grid_size - 1])
            self.state = GridWorldState(x=x, y=y)
        else:
            self.state = GridWorldState()

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

    def sample_state(self):
        pass

    def sample_action(self, args: dict = None):
        return np.random.choice(ACTIONS)

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
        plt.plot(self.goal_pos.x, self.goal_pos.y, "o", color="green", label="GOAL")

        # agent
        plt.plot(self.state.agent_pos.x, self.state.agent_pos.y, "o", color="blue", label="AGENT")

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
        frames = [
            Image.open(image) for image in sorted(glob.glob(f"{name}/*.jpeg"))
        ]
        frame_one = frames[0]
        frame_one.save(
            f"{self.dir}/GridWorldGif_{self.episode_conuter}.gif",
            format="GIF", append_images=frames, save_all=True,
            duration=self.horizon * 100, loop=0
        )

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
        elif env_type == "rooms":
            # TODO: implement
            pass
        elif env_type == "U":
            # there is a U circling around the goal
            self.forbidden_coordinates.append(Position(x=self.goal_pos.x + 1, y=self.goal_pos.y))
            self.forbidden_coordinates.append(Position(x=self.goal_pos.x + 1, y=self.goal_pos.y + 1))
            self.forbidden_coordinates.append(Position(x=self.goal_pos.x + 1, y=self.goal_pos.y - 1))
            self.forbidden_coordinates.append(Position(x=self.goal_pos.x, y=self.goal_pos.y - 1))
            self.forbidden_coordinates.append(Position(x=self.goal_pos.x - 1, y=self.goal_pos.y - 1))
            self.forbidden_coordinates.append(Position(x=self.goal_pos.x - 1, y=self.goal_pos.y))
            self.forbidden_coordinates.append(Position(x=self.goal_pos.x - 1, y=self.goal_pos.y + 1))
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
            if self.ding_flag:
                self.grid_map[:, :] = 0
            else:
                self.grid_map[:, :] = -1
        else:
            pass
        if self.ding_flag:
            self.grid_map[self.goal_pos.x, self.goal_pos.y] = 1
        else:
            self.grid_map[self.goal_pos.x, self.goal_pos.y] = 0

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
            goal_tol: float = 0, obstacles_strict_flag: bool =  False,
            use_costs: bool = False, sampling_strategy: str = "random", sampling_args: dict = None,
            threshold: float = 0.1
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

            use_costs (bool): a flag telling to use cost function. In the case
            in which the user decides to use costs, then the step function will
            return as additional output the list of cost values. Default to
            "False".

            sampling_strategy (str): the strategy used to sample states.
            Default to "random".

            sampling_args (dict): dictionary containing the arguments to sample
            the states.
        """
        # Super class initialization
        super().__init__(horizon=horizon, gamma=gamma,verbose=verbose)  # self.horizon, self.gamma, self.time

        # Map initialization
        self.grid_size = grid_size
        self.goal_pos = GridWorldState(x=self.grid_size / 2, y=self.grid_size / 2)
        self.goal = self.goal_area()
        self.threshold = threshold
        self.pacman = pacman
        self.epsilon = goal_tol

        # State initialization
        err_msg = f"[GridWorld] the strategy {sampling_strategy} is not implemented."
        assert sampling_strategy in ["random", "sphere"], err_msg
        self.sampling_strategy = sampling_strategy
        self.sampling_args = sampling_args
        # print(self.sampling_args)
        self.init_state = init_state
        if init_state is not None:
            self.state = GridWorldState(x=init_state[0], y=init_state[1])
        else:
            # self.state = self.sample_state(strategy="init", args=copy.deepcopy(self.sampling_args))[0]
            self.state = GridWorldState(x = 0.5, y = 0.5)

        self.state_dim = 2
        self.action_dim = 2

        # Reward
        assert reward_type in LEGAL_REWARDS, "[ERROR] Illegal reward type."
        self.reward_type = reward_type

        # Obstacles
        self.obstacles = design_u_obstacle(grid_size, 0.5)
        self.obstacles_strict_flag = obstacles_strict_flag
        self.use_costs  =self.with_costs = use_costs
        self.n_costs = self.how_many_costs = 1

        # Saving parameters
        self.render = render
        self.dir = dir
        self.episode_counter = 1
        if self.render and self.dir is None:
            print("[Error] No directory provided.")
            quit(-1)
        self.history = {"0": {
            "pos_x": round(self.state.agent_pos.x, 1),
            "pos_y": round(self.state.agent_pos.y, 1)
        }}
        if self.render:
            self.save_single_image()

    def goal_area(self):
        goal_center_x = self.goal_pos.x
        goal_center_y = self.goal_pos.y
        goal_width = 1.0 # Width of the rectangular goal region
        goal_height = 0.5  # Height of the rectangular goal region

        # Calculate the four corner vertices of the rectangular goal region, with goal_pos at the top center
        top_left = Position(goal_center_x - goal_width / 2, goal_center_y)
        top_right = Position(goal_center_x + goal_width / 2, goal_center_y)
        bottom_left = Position(goal_center_x - goal_width / 2, goal_center_y - goal_height)
        bottom_right = Position(goal_center_x + goal_width / 2, goal_center_y - goal_height)

        # Ensure the vertices are ordered counterclockwise
        goal = [bottom_left, top_left, top_right, bottom_right]

        return goal

    def sample_state(self, strategy: str = "random", args=None):
        if args is None:
            args = {}
        if strategy == "random":
            return self.sample_random_state(**args)
        elif strategy == "sphere":
            return self.sample_state_from_sphere(**args)
        elif strategy == "init":
            x = np.random.uniform(self.threshold, self.grid_size - self.threshold)
            y = np.random.uniform(self.threshold, self.grid_size / 5 - self.threshold)

            return [GridWorldState(x=x, y=y)]
        else:
            err_msg = f"[GridWorld] strategy {strategy} is not implemented to sample states!"
            raise NotImplementedError(err_msg)

    def sample_state_from_sphere(
            self, n_samples: int = 1,
            radius: float = 0,
            noise: float = 0,
            left_lim: float = 0,
            right_lim: float = 2*np.pi,
            density: int = 100
    ) -> list:
        """
        Summary:
            function to sample states over a circle centered in the goal
        Args:
             n_samples (int): how many points are required.
             Default to 1.

             radius (float): radius of the circle centered in the goal from
             which it is requested a sampling.
             Default to 0.

             noise (float): noise to apply over the radius when sampling.
             Default to 0.

             left_lim (float): lower limit of the angle space in radians.
             Default to 0.

             right_lim (float): upper limit of the angle space in radians.
             Default to 2π.

             density (int): density of the point cloud from which sampling.
             Default to 100.
        Returns:
            list of sampled states
        """
        # checks
        radius = max(radius, 0)
        noise = max(noise, 0)
        density = max(density, 1)
        left_lim = max(left_lim, 0)
        right_lim = min(right_lim, 2*np.pi)

        # sample noises
        noises = np.random.normal(0, noise, n_samples)

        # adjust radii
        radii = radius * np.ones(n_samples) + noises

        # angles
        angles_space = np.linspace(left_lim, right_lim, density, endpoint=False)
        angles = angles_space[np.random.choice(range(density), n_samples, replace=False)]

        # compute coordinates
        x_coordinates = radii * np.cos(angles) + self.goal_pos.x * np.ones(n_samples)
        y_coordinates = radii * np.sin(angles) + self.goal_pos.y * np.ones(n_samples)
        res = []
        for i in range(n_samples):
            res.append(GridWorldState(x=x_coordinates[i], y=y_coordinates[i]))
        return res

    def sample_random_state(self, n_samples: int = 1):
        """
        Summary:
            this function samples a desired number of initial states and return
            them as a list.
        Args:
            n_samples: how many initial states are to be drawn.

        Returns:
            list of initial states
        """
        space = np.linspace(start=0, stop=self.grid_size, num=100)

        if n_samples <= 1:
            """Just one initial state"""
            state = GridWorldState(x=np.random.choice(space),
                                   y=np.random.choice(space))
            return [state]
        else:
            """More initial states"""
            x_coordinates = np.random.choice(a=space, size=n_samples,
                                             replace=False)
            y_coordinates = np.random.choice(a=space, size=n_samples,
                                             replace=False)
            states = []
            for i in range(n_samples):
                states.append(GridWorldState(x=x_coordinates[i],y=y_coordinates[i]))
            return states

    def step(self, action: GWContMove = None) -> tuple:
        """
        Summary: function implementing a step of the environment.
        Args:
            action (GWContMove): the action to apply to the environment.
        Returns:
            new position, reward and absorbing flag.
            (Occasionally, when self.use_costs, it returns also the np.array of
            costs).
        """

        self.time += 1
        outside_flag = False

        # Compute the next state
        new_pos = deepcopy(self.state)

        new_pos.x = action.radius * np.cos(np.deg2rad(action.theta)) + self.state.agent_pos.x
        new_pos.y = action.radius * np.sin(np.deg2rad(action.theta)) + self.state.agent_pos.y

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
        isNewAbs = self.is_absorbing(new_pos.agent_pos)
        forbidden = 0
        for obs in self.obstacles:
            if obs.is_in(pos=new_pos):
                forbidden += 1
                break

        # Compute the reward
        reward = self.reward(position=new_pos, outside=outside_flag, forbidden=forbidden)

        if (not forbidden and not isAbs) or (forbidden and not self.obstacles_strict_flag):
            # Update state
            self.state.agent_pos = deepcopy(new_pos)
        else:
            # Reload old state
            new_pos = deepcopy(self.state.agent_pos)

        # Save results
        self.history[str(self.time)] = {
            "pos_x": round(new_pos.x, 1),
            "pos_y": round(new_pos.y, 1)
        }
        if self.render:
            self.save_single_image()

        # Costs computation
        if self.use_costs:
            info = dict(costs = self.cost_obstacle(forbidden))

            if isNewAbs:
                # self.reset()
                return new_pos, (self.horizon - self.time) * reward, True,  info
            else:
                return new_pos, reward, isNewAbs, info
        # No costs have to be included
        else:
            if isNewAbs:
                return new_pos, (self.horizon - self.time) * reward, True
            else:
                return new_pos, reward, isNewAbs

    def reset(self, state = None)-> None:
        if state is not None:
            self.state = state
        else:

            # self.time init
            super().reset()

            # State initialization
            if self.init_state is not None:
                self.state = GridWorldState(x=self.init_state[0], y=self.init_state[1])
            else:
                # self.state = self.sample_state(strategy="init",args=copy.deepcopy(self.sampling_args))[0]
                self.state = GridWorldState(x = 0.5, y = 0.5)

        # Save results
        if self.dir is not None:
            self.save_history()
            if self.render:
                self.save_gif()

        self.episode_counter += 1
        # History
        self.history = {'0': {
            "pos_x": round(self.state.agent_pos.x, 1),
            "pos_y": round(self.state.agent_pos.y, 1),
        }}

        if self.render:
            self.save_single_image()

        self.time = 0

    def reward(self, position, outside: int = False, forbidden: int = False) -> float:
        if forbidden and not self.use_costs:
            # If the agent is in a forbidden zone, punish it with a super
            # negative reward
            return -(self.grid_size ** 2) * forbidden
        else:
            # Distance computation
            dist = (position.x -self.goal_pos.x) ** 2
            dist += (position.y -self.goal_pos.y) ** 2
            dist = np.sqrt(dist)

            # Penalization if the policy said to go outside
            dist += self.grid_size if outside else 0

            # Give reward
            if self.is_absorbing(position):
                #print(f'[GridWorld] Goal reached.')
                return 0
            if self.reward_type == "linear":
                return -dist
            elif self.reward_type == "sparse":
                if forbidden or outside:
                    return -self.grid_size
                return -1
            else:
                raise NotImplementedError

    def is_absorbing(self, position: Position):
        """
        Check if the agent's position is inside the goal area defined by a rectangle.
        """
        bottom_left, top_left, top_right, bottom_right = self.goal

        # Ensure that you use the correct bounds from the rectangle corners
        within_x_bounds = bottom_left.x <= position.x <= top_right.x
        within_y_bounds = bottom_left.y <= position.y <= top_left.y

        return within_x_bounds and within_y_bounds

    def save_history(self) -> None:
        """Save json of the history"""
        name = self.dir + f"/GridWorldContResults_{self.episode_counter}.json"
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with io.open(name, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.history, ensure_ascii=False, indent=4))
            f.close()

    def save_single_image(self) -> None:
        """Save an image of the current state of the environment"""
        # Clear plot
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_xticks(np.arange(self.grid_size + 1))
        ax.set_yticks(np.arange(self.grid_size + 1))

        # Goal position (goal center point)
        ax.plot(self.goal_pos.x, self.goal_pos.y, "o", color="green", label="GOAL")

        # Agent position
        ax.plot(self.state.agent_pos.x, self.state.agent_pos.y, "o", color="blue", label="AGENT")

        # Plot walls (obstacles)
        i = 0
        for obs in self.obstacles:
            if obs.type == "square":
                x = obs.features["p1"].x
                y = obs.features["p1"].y
                w = obs.features["p2"].x - obs.features["p1"].x
                h = obs.features["p4"].y - obs.features["p1"].y
                if i == 0:
                    ax.add_patch(plt.Rectangle((x, y), w, h, alpha=0.3, color="red", label="obstacle"))
                    i += 1
                else:
                    ax.add_patch(plt.Rectangle((x, y), w, h, alpha=0.3, color="red"))
            elif obs.type == "circle":
                x = obs.features["center"].x
                y = obs.features["center"].y
                r = obs.features["radius"]
                ax.add_patch(plt.Circle((x, y), r, alpha=0.3, color="orange", label="obstacle"))

        # Plot the goal area as a filled rectangle
        if hasattr(self, 'goal') and len(self.goal) == 4:
            goal_points = [[p.x, p.y] for p in self.goal]
            goal_polygon = plt.Polygon(goal_points, closed=True, edgecolor="green", facecolor="lime", alpha=0.3,
                                       label="Goal Area")
            ax.add_patch(goal_polygon)

        # Plot positions and settings
        plt.grid()
        plt.legend(loc="best")
        plt.title(f"GridWorld EPISODE {self.episode_counter} FRAME {self.time}")

        # Construct file name for saving
        if self.time < 10:
            name = self.dir + f"/frames_{self.episode_counter}/GridWorldFrame_{self.episode_counter}_00{self.time}.jpeg"
        elif self.time < 100:
            name = self.dir + f"/frames_{self.episode_counter}/GridWorldFrame_{self.episode_counter}_0{self.time}.jpeg"
        else:
            name = self.dir + f"/frames_{self.episode_counter}/GridWorldFrame_{self.episode_counter}_{self.time}.jpeg"

        # Ensure directory exists
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        # Save the figure
        plt.savefig(name, format="jpeg", dpi=300)
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

    def cost_obstacle(self, forbidden: int = 0):
        """When the agent is in a forbidden zone, it has a cost."""
        return forbidden

    def cost_boundary(self, state: Position =  None):
        """When the agent is near to the boundary, it has a cost."""

        if state is None:
            state = self.state.agent_pos

        cond_near_x = (state.x <= self.threshold) or (self.grid_size - state.x <= self.threshold)
        cond_near_y = (state.y <= self.threshold) or (self.grid_size - state.y <= self.threshold)

        # print(f'[GridWorld] Agent is near to the boundary: {cond_near_x or cond_near_y} at position {self.state.agent_pos.x, self.state.agent_pos.y}')
        if cond_near_x or cond_near_y:
            # return self.grid_size / 2
            return 1
        else:
            return 0

    def sample_action(self, args: dict = None):
        return GWContMove(radius=1, theta=np.random.uniform(0, 2 * np.pi))

    def set_state(self, state):
        x = state.x
        y = state.y
        self.state = GridWorldState(x=x, y=y)


def main():
    horizon = 100
    gamma = 0.99

    env = GridWorldEnvCont(
        horizon=horizon,
        gamma=gamma,
        grid_size=7,
        reward_type="linear",
        render=True,
        dir="/Users/leonardo/Desktop/Thesis/MagicRL/images",
    )

    # env.reset()
    # env.save_single_image()


if __name__ == "__main__":
    main()
