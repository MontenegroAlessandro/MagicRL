"""
Implementation of the LQR Environment.
Import from MushroomRL
"""
import numpy as np
import scipy
from abc import ABC
from numbers import Number
from envs.base_env import BaseEnv
from envs.utils import StateBoundsIdx, ActionBoundsIdx


class LQR(BaseEnv, ABC):
    def __init__(
            self,
            A: np.ndarray = None,
            B: np.ndarray = None,
            Q: np.ndarray = None,
            R: np.ndarray = None,
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
        Summary:
            Initialization of the class.

        Args:
            A (np.ndarray): the state dynamics matrix;
            B (np.ndarray): the action dynamics matrix;
            Q (np.ndarray): reward weight matrix for state;
            R (np.ndarray): reward weight matrix for action;
            max_pos (float, np.inf): maximum value of the state;
            max_action (float, np.inf): maximum value of the action;
            random_init (bool, False): start from a random state;
            episodic (bool, False): end the episode when the state goes over
            the threshold;
            gamma (float, 0.9): discount factor;
            horizon (int, 50): horizon of the mdp;
            dt (float, 0.1): the control timestep of the environment.
        """
        super().__init__(horizon=horizon, gamma=gamma)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self._max_pos = max_pos
        self._max_action = max_action
        self._episodic = episodic
        self.random_init = random_init
        self._initial_state = initial_state
        self.dt = dt

        # State bounds
        high_x = self._max_pos * np.ones(A.shape[0])
        low_x = -high_x
        self.state_bounds = [low_x, high_x]
        self.state_dim = len(self.A)

        # Action bounds
        high_u = self._max_action * np.ones(B.shape[1])
        low_u = -high_u
        self.action_bounds = [low_u, high_u]
        self.action_dim = len(self.B[0])

        self.state = None
        self.K = self.get_optimal_K()
        # self.K = self.computeOptimalK()

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
        Factory method that generates a LQR with identity dynamics and
        symmetric reward matrices.

        Args:
            dimensions (int): number of state-action dimensions;
            s_dim (int): number of state dimensions;
            a_dim (int): number of action dimensions;
            max_pos (float, np.inf): maximum value of the state;
            max_action (float, np.inf): maximum value of the action;
            eps (double, .1): reward matrix weights specifier;
            index (int, 0): selector for the principal state;
            scale (float, 1.0): scaling factor for the reward function;
            random_init (bool, False): start from a random state;
            episodic (bool, False): end the episode when the state goes over the threshold;
            gamma (float, .9): discount factor;
            horizon (int, 50): horizon of the mdp;
            initial_state: the fixed initial state of the environment when reset.
        """
        assert dimensions is not None or (s_dim is not None and a_dim is not None)

        if s_dim is None or a_dim is None:
            s_dim = dimensions
            a_dim = dimensions
        A = scale_matrix * np.eye(s_dim)
        """if s_dim > a_dim == 1:
            B = scale_matrix * np.ones((s_dim, a_dim)) / np.sqrt(s_dim)
        else:
            B = scale_matrix * np.eye(s_dim, a_dim)"""
        B = scale_matrix * np.eye(s_dim, a_dim)
        Q = eps * np.eye(s_dim) * scale
        R = (1. - eps) * np.eye(a_dim) * scale

        Q[index, index] = (1. - eps) * scale
        R[index, index] = eps * scale

        return LQR(A, B, Q, R, max_pos, max_action, random_init, episodic, gamma, horizon, initial_state)

    def reset(self, state=None):
        if state is None:
            if self.random_init:
                self.state = np.clip(
                    np.random.uniform(-3, 3, size=self.A.shape[0]),
                    self.state_bounds[StateBoundsIdx.lb],
                    self.state_bounds[StateBoundsIdx.ub]
                )
            elif self._initial_state is not None:
                self.state = self._initial_state
            else:
                init_value = .9 * self._max_pos if np.isfinite(
                    self._max_pos) else 10
                self.state = init_value * np.ones(self.A.shape[0])
        else:
            self.state = state

        return self.state

    def step(self, action):
        x = self.state
        u = np.clip(
            action,
            self.action_bounds[ActionBoundsIdx.lb],
            self.action_bounds[ActionBoundsIdx.ub]
        )

        reward = -(x.dot(self.Q).dot(x) + u.dot(self.R).dot(u))
        self.state = self.A.dot(x) + self.B.dot(u)

        absorbing = False

        if np.any(np.abs(self.state) > self._max_pos):
            if self._episodic:
                reward = -self._max_pos ** 2 * 10
                absorbing = True
            else:
                self.state = np.clip(
                    self.state,
                    self.state_bounds[StateBoundsIdx.lb],
                    self.state_bounds[StateBoundsIdx.ub]
                )

        return self.state, reward, absorbing, {}

    def sample_state(self, args: dict = None):
        NotImplementedError("[LQR] Ops, not implemented yet!")

    def sample_optimal_action(self, state):
        return np.array(self.K @ state)[0]

    def get_optimal_K(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (a = K * s).

        Returns:
            K (matrix): the optimal controller

        """
        X = np.matrix(scipy.linalg.solve_discrete_are(self.A, self.B, self.Q, self.R))
        K = - np.matrix(scipy.linalg.inv(self.B.T * X * self.B + self.R) * (self.B.T * X * self.A))
        return K

    def grad_deterministic_theta(self, theta):
        """
        Deterministic Policy gradient (wrt theta) of linear policy with mean theta.
        Scalar case only
        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if not np.array_equal(self.A, I) or not np.array_equal(self.B, I):
            raise NotImplementedError
        if not isinstance(theta, Number):
            raise NotImplementedError
        theta = np.array(theta).item()

        den = 1 - self.gamma * (1 + 2 * theta + theta ** 2)
        dePdeK = 2 * (theta * self.R / den + self.gamma * (self.Q + theta ** 2 * self.R) * (1 + theta) / den ** 2)
        return (- dePdeK * (self.state_bounds[StateBoundsIdx.ub] ** 2 / 3)).item()

    def _computeP2(self, K):
        """
        This function computes the Riccati equation associated to the LQG
        problem.
        Args:
            K (matrix): the matrix associated to the linear controller a = K s

        Returns:
            P (matrix): the Riccati Matrix

        """
        I = np.eye(self.Q.shape[0], self.Q.shape[1])
        if np.array_equal(self.A, I) and np.array_equal(self.B, I):
            P = (self.Q + np.dot(K.T, np.dot(self.R, K))) / (I - self.gamma *(I + 2 * K + K ** 2))
        else:
            tolerance = 0.0001
            converged = False
            P = np.eye(self.Q.shape[0], self.Q.shape[1])
            while not converged:
                Pnew = (self.Q +
                        self.gamma * np.dot(self.A.T, np.dot(P, self.A)) +
                        self.gamma * np.dot(K.T, np.dot(self.B.T, np.dot(P, self.A))) +
                        self.gamma * np.dot(self.A.T, np.dot(P, np.dot(self.B, K))) +
                        self.gamma * np.dot(K.T, np.dot(self.B.T, np.dot(P, np.dot(self.B, K)))) +
                        np.dot(K.T, np.dot(self.R, K)))
                converged = np.max(np.abs(P - Pnew)) < tolerance
                P = Pnew
        return P

    def computeOptimalK(self):
        """
        This function computes the optimal linear controller associated to the
        LQG problem (a = K * s).

        Returns:
            K (matrix): the optimal controller

        """
        P = np.eye(self.Q.shape[0], self.Q.shape[1])
        for i in range(100):
            K = -self.gamma * np.dot(np.linalg.inv(
                self.R + self.gamma * (np.dot(self.B.T, np.dot(P, self.B)))),
                np.dot(self.B.T, np.dot(P, self.A)))
            P = self._computeP2(K)
        K = -self.gamma * np.dot(np.linalg.inv(self.R + self.gamma * (np.dot(self.B.T, np.dot(P, self.B)))), np.dot(self.B.T, np.dot(P, self.A)))
        return K
