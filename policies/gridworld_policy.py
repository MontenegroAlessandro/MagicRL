"""Implementation of a parametric policy for the continuous version of
Grid World."""
import copy

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
                        std_min: float = 1e-5,
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
        self.parameters = np.array(thetas)
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
        mean_theta = self.parameters @ state
        theta = np.pi * np.tanh(mean_theta)
        # print(f'Rad theta: {theta}, Grad theta: {mean_theta}')

        if self.alg == 'cpg':
            # theta = np.random.vonmises(theta, 1/self.std_dev)
            theta = np.random.normal(theta, self.std_dev)
            # print(f'Original theta: {theta_prime}, New theta: {theta}')
            # theta = np.pi * np.tanh(np.random.normal(mean_theta, self.std_dev))
            # print(f'Before noise: {np.pi * np.tanh(mean_theta)}, After noise {theta}')

        # return GWContAction(radius=radius, theta=np.rad2deg(theta))
        return GWContAction(radius=radius, theta=theta)

    def set_parameters(self, thetas):
        self.parameters = copy.deepcopy(np.array(thetas))

    def get_parameters(self):
        return self.parameters

    def compute_score(self, state, action):
        """
        Computes the score in radians.
        """
        if self.std_dev == 0:
            return super().compute_score(state, action)

        def sech(x):
            return 1 / np.cosh(x)
        """
        # Ensure action.theta is in radians
        action_rad = np.deg2rad(action.theta)

        # Compute the deviation in radians
        mean_theta = self.parameters @ state
        mean_theta_rad = np.deg2rad(mean_theta)
        action_deviation = action_rad - mean_theta_rad

        # Compute the Gaussian term
        gaussian_term = np.exp(-(action_deviation ** 2) / (self.std_dev ** 2))

        # Normalize the Gaussian term
        corrected_gaussian_term = gaussian_term / (np.sqrt(2 * np.pi) * self.std_dev)

        # Compute the sech term
        sech_term = sech(corrected_gaussian_term) ** 2

        # Compute numerator and denominator
        numerator = np.sqrt(2) * gaussian_term * sech_term * action_deviation
        denominator = np.sqrt(np.pi) * (self.std_dev ** 3) * np.tanh(corrected_gaussian_term)

        # Add a small epsilon to denominator for numerical stability
        epsilon = 1e-8
        score = numerator / (denominator + epsilon)

        print(f'Action: {action}, Parameters @  state: {self.parametres @ state}, Gaussian term: {gaussian_term},'
              f' Corrected gaussian term: {corrected_gaussian_term}, Sech term: {sech_term}, Action deviation: {action_deviation}, Score {score}'
              f' Numerator: {numerator}, Denominator: {denominator}')
        return score * state
        """
        state = np.array(state)

        mean_theta = self.parameters @ state

        # action_deviation = np.deg2rad(action.theta) - np.pi * np.tanh(mean_theta)
        action_deviation = action.theta - np.pi * np.tanh(mean_theta)

        grad_theta = (sech(mean_theta) ** 2)

        return np.pi * action_deviation * grad_theta * state / (self.std_dev ** 2)


    def reduce_exploration(self):
        self.std_dev = np.clip(self.std_dev - self.std_decay, self.std_min, np.inf)

    def diff(self, state):
        raise NotImplementedError