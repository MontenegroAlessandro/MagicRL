"""Implementation of a parametric policy for the continuous version of
Grid World."""
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

    def __init__(
                        self, thetas: np.array = None,
                        std_dev: float = 0.1,
                        std_decay: float = 0,
                        std_min: float = 1e-4,
                        dim_state: int = 1,
                        alg: str = None
                         ) -> None:
        """
        Args:
            thetas (list): Parameter initialization for the policy "[, Ω]"
            dim_state (int): size of the state (features)
        """
        super().__init__()
        self.dim_state = dim_state
        # self.thetas = np.array(thetas[:dim_state])
        # self.omegas = np.array(thetas[dim_state:])
        self.thetas = np.array(thetas)
        self.tot_params = len(thetas)
        self.std_dev = std_dev
        self.std_decay = std_decay
        self.std_min = std_min
        self.alg = alg

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
        # radius = 1 / (1 + np.exp(-self.omegas.T @ state))
        radius = 0.1
        mean_theta = self.thetas.T @ state
        theta = np.pi * np.tanh(mean_theta)

        if self.alg == 'cpg':
            theta = np.random.normal(theta, self.std_dev)

        return GWContAction(radius=radius, theta=np.rad2deg(theta))

    def set_parameters(self, thetas):
        self.thetas = np.array(thetas)

    def get_parameters(self):
        return self.thetas

    def compute_score(self, state, action):
        if self.std_dev == 0:
            return super().compute_score(state, action)

        def sech(x):
            return 1 / np.cosh(x)

        state = np.array(state)
        mean_theta = self.thetas.T @ state

        action_deviation = action.theta - np.pi * np.tanh(mean_theta)

        grad_theta = (sech(mean_theta) ** 2)

        return  action_deviation * grad_theta * state / (self.std_dev ** 2)

    def reduce_exploration(self):
        self.std_dev = np.clip(self.std_dev - self.std_decay, self.std_min, np.inf)

    def diff(self, state):
        raise NotImplementedError