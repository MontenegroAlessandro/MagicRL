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
