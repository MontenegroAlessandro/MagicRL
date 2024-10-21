"""
Implementation of the LQR Environment with costs.
"""
import copy
import numpy as np
from envs.lqr import LQR, LQRDiscrete
from envs.utils import ActionBoundsIdx, StateBoundsIdx


class CostLQR(LQR):
    def __init__(
            self,
            A: np.ndarray=None,
            B: np.ndarray=None,
            Q: np.ndarray=None,
            R: np.ndarray=None,
            max_pos=np.inf,
            max_action=np.inf,
            random_init=False,
            episodic=False,
            gamma=0.9,
            horizon=50,
            initial_state=None,
            dt=0.1
    ):
        """
        See the parameters in the original LQR environment.

        Note that this environment is such that:
            s_{t+1} = A s_{t} + B a_{t}
            r_{t} = -s_{t}.T Q s_{t}
            c_{t} = -a_{t}.T R a_{t}
        where "c" represents the instantaneous cost.
        """
        # Super class initialization
        super(CostLQR, self).__init__(
            A=A, B=B, Q=Q, R=None, max_pos=max_pos, max_action=max_action, random_init=random_init,
            episodic=episodic, gamma=gamma, horizon=horizon, initial_state=initial_state, dt=dt
        )

        # Define the cost matrix (the one related to the action)
        self.C = R

        # Define fields for the costs management
        self.with_costs = True
        self.how_many_costs = 1

    @staticmethod
    def generate(
            dimensions=None,
            s_dim=None,
            a_dim=None,
            max_pos=np.inf,
            max_action=np.inf,
            scale_matrix=1,
            eps=0.1,
            index=0,
            scale=1.0,
            random_init=True,
            episodic=False,
            gamma=.99,
            horizon=50,
            initial_state=None
    ):
        """
        See the meaning of the parameters in the original LQR environment.
        """
        assert dimensions is not None or (s_dim is not None and a_dim is not None)

        if s_dim is None or a_dim is None:
            s_dim = dimensions
            a_dim = dimensions

        A = scale_matrix * np.eye(s_dim)
        B = scale_matrix * np.eye(s_dim, a_dim)
        Q = eps * np.eye(s_dim) * scale
        R = (1. - eps) * np.eye(a_dim) * scale

        Q[index, index] = (1. - eps) * scale
        R[index, index] = eps * scale

        return CostLQR(
            A, B, Q, R, max_pos, max_action, random_init, episodic, gamma, horizon, initial_state
        )

    def step(self, action):
        """
        This is the modified version of the step, which is returning:
            i. next state
            ii. reward of the step
            iii. absorbing flag
            iv. additional info
            v. an array of the costs we have in the environment
        """
        # Preprocess the action
        u = np.clip(
            action,
            self.action_bounds[ActionBoundsIdx.lb],
            self.action_bounds[ActionBoundsIdx.ub]
        )

        # Compute the canonical step
        obs, reward, done, info = super().step(action=action)

        # Compute the costs (here just one cost)
        cost = u.dot(self.C).dot(u)

        # Put the cost information into the info dictionary
        info["costs"] = copy.deepcopy(np.array([cost], dtype=np.float64))

        return obs, reward, done, info

# Discrete version of the CostLQR
class CostLQRDiscrete(LQRDiscrete):
    def __init__(
            self,
            A: np.ndarray=None,
            B: np.ndarray=None,
            Q: np.ndarray=None,
            R: np.ndarray=None,
            max_pos=np.inf,
            max_action=np.inf,
            random_init=False,
            episodic=False,
            gamma=0.9,
            horizon=50,
            initial_state=None,
            dt=0.1,
            state_bins: int = 10,
            action_bins: int = 10
    ):
        """
        See the parameters in the original LQR environment.

        Note that this environment is such that:
            s_{t+1} = A s_{t} + B a_{t}
            r_{t} = -s_{t}.T Q s_{t}
            c_{t} = -a_{t}.T R a_{t}
        where "c" represents the instantaneous cost.
        """
        # Super class initialization
        super(CostLQRDiscrete, self).__init__(
            A=A, B=B, Q=Q, R=None, max_pos=max_pos, max_action=max_action, random_init=random_init,
            episodic=episodic, gamma=gamma, horizon=horizon, initial_state=initial_state, dt=dt,
            state_bins=state_bins, action_bins=action_bins
        )

        # Define the cost matrix (the one related to the action)
        self.C = R

        # Define fields for the costs management
        self.with_costs = True
        self.how_many_costs = 1

    @staticmethod
    def generate(
            dimensions=None,
            s_dim=None,
            a_dim=None,
            max_pos=np.inf,
            max_action=np.inf,
            scale_matrix=1,
            eps=0.1,
            index=0,
            scale=1.0,
            random_init=True,
            episodic=False,
            gamma=.99,
            horizon=50,
            initial_state=None,
            state_bins=10,
            action_bins=10
    ):
        """
        See the meaning of the parameters in the original LQR environment.
        """
        assert dimensions is not None or (s_dim is not None and a_dim is not None)

        if s_dim is None or a_dim is None:
            s_dim = dimensions
            a_dim = dimensions

        A = scale_matrix * np.eye(s_dim)
        B = scale_matrix * np.eye(s_dim, a_dim)
        Q = eps * np.eye(s_dim) * scale
        R = (1. - eps) * np.eye(a_dim) * scale

        Q[index, index] = (1. - eps) * scale
        R[index, index] = eps * scale

        return CostLQRDiscrete(
            A, B, Q, R, max_pos, max_action, random_init, episodic, gamma, horizon, initial_state,
            state_bins=state_bins, action_bins=action_bins
        )

    def step(self, action):
        x = self.state
        
        # note that action is the index
        action = copy.deepcopy(self.action_enumeration[action])
        
        # Preprocess the action
        u = np.clip(
            action,
            self.action_bounds[ActionBoundsIdx.lb],
            self.action_bounds[ActionBoundsIdx.ub]
        )
        
        # compute rewards
        reward = -(x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))
        self.state = self.A.dot(x) + self.B.dot(u)
        
        # discretization of the state
        self.state = self.state_map[np.digitize(self.state, self.state_map) - 1]

        absorbing = False

        if np.any(np.abs(self.state) > self.max_pos):
            if self._episodic:
                reward = -self.max_pos ** 2 * 10
                absorbing = True
            else:
                self.state = np.clip(
                    self.state,
                    self.state_bounds[StateBoundsIdx.lb],
                    self.state_bounds[StateBoundsIdx.ub]
                )

        # Compute the costs (here just one cost)
        cost = u.dot(self.C).dot(u)

        # Put the cost information into the info dictionary
        info = dict()
        info["costs"] = copy.deepcopy(np.array([cost], dtype=np.float64))

        return self.state, reward, absorbing, info


class RobotWorld(LQR):
    def __init__(
            self,
            Q: np.ndarray = None,
            R: np.ndarray = None,
            max_pos: float = np.inf,
            max_action: float = np.inf,
            random_init: bool = False,
            episodic: bool = False,
            gamma: float = 0.9,
            horizon: int = 50,
            initial_state: np.ndarray = None,
            dt: float = 0.05,
            tau: float = 0.1,
            range_pos: np.ndarray = np.array([40, 50]),
            range_vel: np.ndarray = np.array([-0.1, 0.1])
    ) -> None:
        """
        Initialize the RobotWorld environment in a way similar to CostLQR.
        """

        # Time step size
        self.dt = dt

        A, B = self.generate_dynamics()

        super(RobotWorld, self).__init__(
            A=A, B=B, Q=Q, R=None, max_pos=max_pos, max_action=max_action, random_init=random_init,
            episodic=episodic, gamma=gamma, horizon=horizon, initial_state=initial_state, dt=dt
        )

        # Define the cost matrix (the one related to the action)
        self.C = R

        # Additional parameters specific to RobotWorld
        self.tau = tau
        self.with_costs = True
        self.how_many_costs = 1

        # random variable
        self.rng = np.random.default_rng()

        # Define the cost matrices
        self.G1 = - np.array([1.0, 1.0, 0.001, 0.001])
        self.G2 = - np.array([0.001, 0.001, 1.0, 1.0])
        self.R1 = - np.array([0.01, 0.01])
        self.R2 = - np.array([0.01, 0.01])

        # State bounds
        self.range_pos = range_pos
        self.range_vel = range_vel

    @staticmethod
    def generate(
            s_dim: int = None,
            a_dim: int = None,
            max_pos: float = np.inf,
            max_action: float = np.inf,
            scale_matrix: float = 1.0,
            eps: float = 0.1,
            index: int = 0,
            scale: float = 1.0,
            random_init: bool = True,
            episodic: bool = False,
            gamma: float = 0.99,
            horizon: int = 50,
            dt: float = 0.05,
            tau: float = 0.1,
            initial_state: np.ndarray = None,
            range_pos: np.ndarray = np.array([40, 50]),
            range_vel: np.ndarray = np.array([-0.1, 0.1])
    ):
        """
        Generate a new RobotWorld environment with random dynamics matrices and cost matrices.

        :param dimensions: The dimension for both the state and action (s_dim and a_dim).
        :param s_dim: Dimension of the state space (optional if dimensions is given).
        :param a_dim: Dimension of the action space (optional if dimensions is given).
        :param max_pos: Maximum position constraint.
        :param max_action: Maximum action constraint.
        :param scale_matrix: Scaling factor for the system matrices A and B.
        :param eps: Scaling factor for cost matrices Q and R.
        :param index: Index for modifying specific values in Q and R.
        :param scale: General scale for Q and R.
        :param random_init: Whether the system should initialize states randomly.
        :param episodic: Whether the environment is episodic.
        :param gamma: Discount factor.
        :param horizon: Planning horizon.
        :param dt: Time step for the dynamics.
        :param tau: Regularization parameter for the reward.
        :param initial_state: Initial state of the environment.
        :return: A new RobotWorld instance with generated dynamics and cost matrices.
        """

        # Generate state cost matrix Q and action cost matrix R
        Q = eps * np.eye(s_dim) * scale
        R = (1.0 - eps) * np.eye(a_dim) * scale

        # Modify specific entries of Q and R
        Q[index, index] = (1.0 - eps) * scale
        R[index, index] = eps * scale

        # Return a new RobotWorld instance with the generated matrices
        return RobotWorld(
            Q=Q, R=R, max_pos=max_pos, max_action=max_action, random_init=random_init,
            episodic=episodic, gamma=gamma, horizon=horizon, dt=dt, tau=tau, initial_state=initial_state,
            range_pos = range_pos, range_vel = range_vel
        )

    def generate_dynamics(self):
        """
        Generate system dynamics matrices A and B based on the time step.

        :param dt: The time step used in the dynamics.
        :return: A and B matrices for the system dynamics.
        """
        A = np.array(
            [
                [1., 0, self.dt, 0],
                [0, 1., 0, self.dt],
                [0, 0, 1., 0],
                [0, 0, 0, 1.],
            ]
        )

        B = np.array(
            [
                [self.dt ** 2 / 2, 0.0],
                [0.0, self.dt ** 2 / 2],
                [self.dt, 0.0],
                [0.0, self.dt],
            ]
        )

        return A, B

    def reset(self, state=None):
        """
        Reset the environment to a random or predefined initial state.

        :param n_samples: Number of samples (initial states) to generate.
        :return: The initial state of the environment.
        """
        if state is not None:
            self.state = state
        else:
            s = np.array([
                self.rng.uniform(self.range_pos[0], self.range_pos[1]),
                self.rng.uniform(self.range_pos[0], self.range_pos[1]),
                self.rng.uniform(self.range_vel[0], self.range_vel[1]),
                self.rng.uniform(self.range_vel[0], self.range_vel[1]),
            ])

            self.state = s
        return self.state

    def generate_noise(self, size):
        """
        Generate random noise for the system.

        :param size: Shape of the noise to be generated.
        :return: An array containing the generated noise.
        """
        return self.rng.normal(
            scale=np.array([
                1.0,  # noise for position
                1.0,  # noise for position
                1.0,  # noise for velocity
                1.0  # noise for velocity
            ]) * self.dt,  # scale the noise by the time step
            size=size
        )

    def step(self, action):
        """
        Perform a step in the environment by applying the action 'u'.

        :param u: The action applied in the environment.
        :return: The next state, reward, done flag, additional info, and costs.
        """
        # Preprocess the action (clipping it to the action bounds)
        clipped_action = np.clip(
            action,
            self.action_bounds[ActionBoundsIdx.lb],
            self.action_bounds[ActionBoundsIdx.ub]
        )

        # Apply state transition dynamics
        s_noiseless = self.state @ self.A.T + clipped_action @ self.B.T

        noise = self.generate_noise(self.state.shape)
        self.state = s_noiseless + noise

        # Compute reward
        reward = (np.abs(self.state).T @ self.G1).sum(axis=0) + (np.abs(clipped_action).T @ self.R1).sum(axis=0)
        reward += - (self.tau / 2) * (clipped_action.T @ clipped_action).sum(axis=0)

        # Compute cost
        cost = ((self.state ** 2) * self.G2).sum(axis=0) + (self.tau / 2) + ((clipped_action ** 2) * self.R2).sum(axis=0)

        # Info dictionary with costs
        info = {"costs": cost}

        # set done flag only to keep the interface of the step() function
        done = False

        # Return the next state, reward, done flag, info, and costs
        return self.state, reward, done, info
