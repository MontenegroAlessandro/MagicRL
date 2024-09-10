"""Implementation of a Gaussian Policy"""
import timeit

# Libraries
from policies.JAX_policies.base_policy_jax import BasePolicyJAX
from abc import ABC
import numpy as np
import copy
import jax.numpy as jnp
from jax import jacfwd, jit

import jax
jax.config.update("jax_enable_x64", True)

class LinearGaussianPolicyJAX(BasePolicyJAX, ABC):
    """
    Implementation of a Gaussian Policy which is linear in the state.
    Thus, the mean will be: parameters @ state.
    The standard deviation is fixed and is defined by the user.
    """
    def __init__(
            self, parameters: np.array = None,
            std_dev: float = 0.1,
            std_decay: float = 0,
            std_min: float = 1e-4,
            dim_state: int = 1,
            dim_action: int = 1,
            multi_linear: bool = False

    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[GaussPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        err_msg = "[GaussPolicy] standard deviation is negative!"
        assert std_dev > 0, err_msg
        self.std_dev = std_dev

        # Additional attributes
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.tot_params = dim_action * dim_state
        self.multi_linear = multi_linear
        self.std_decay = std_decay
        self.std_min = std_min

        self.jacobian = None

        return

    def draw_action(self, state) -> np.array:
        if len(state) != self.dim_state:
            err_msg = "[GaussPolicy] the state has not the same dimension of the parameter vector:"
            err_msg += f"\n{len(state)} vs. {self.dim_state}"
            raise ValueError(err_msg)

        mean = np.array(self.parameters @ state, dtype=np.float64)
        action = np.array(np.random.normal(mean, self.std_dev), dtype=np.float64)
        return action

    def reduce_exploration(self):
        self.std_dev = np.clip(self.std_dev - self.std_decay, self.std_min, np.inf)

    def set_parameters(self, thetas) -> None:
        if not self.multi_linear:
            self.parameters = copy.deepcopy(thetas)
        else:
            self.parameters = np.array(np.split(thetas, self.dim_action))

    def get_parameters(self):
        return self.parameters

    def _log_policy(self, parameters, state, action):
        if self.multi_linear:
            log_pol = (action - jnp.dot(parameters, state))[:, np.newaxis]
        else:
            log_pol =action - jnp.dot(parameters, state)
        return (log_pol ** 2 / (2 * self.std_dev ** 2))[0]

    """
    def compile_jacobian(self):
        log_policy = jit(self._log_policy)
        self.jacobian = jit(jacfwd(log_policy, argnums=0))

        return
    """

    def compute_score(self, state, action):
        return super().compute_score(state, action)

    def diff(self, state):
        raise NotImplementedError

