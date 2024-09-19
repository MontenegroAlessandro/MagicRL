"""Base Policy class implementation"""
# Libraries
from abc import ABC, abstractmethod
from jax import vmap, jacfwd, jit
import jax.numpy as jnp
import numpy as np



# Class
class BasePolicyJAX(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dim_state = None
        self.dim_action = None
        self.tot_params = None
        self.multi_linear = None
        self.parameters = None

    @abstractmethod
    def draw_action(self, state):
        pass

    @abstractmethod
    def set_parameters(self, thetas):
        pass

    def compile_jacobian(self):
        self.jacobian = jit(jacfwd(self._log_policy, argnums=0))
        self.state_action_score_jit = jit(self.state_action_score)
        self.parameter_score_jit = jit(self.parameter_score)

        return

    def _log_policy(self, parameters, state, action):
        pass

    # Define the scoring function to map over the batch of states and actions
    def state_action_score(self, state, action):
        # Now vmap over parameters instead of over states/actions
        scores = vmap(lambda params: (-1) * self.jacobian(params, state, action))(self.parameters)

        # Apply ravel to each score vector
        return jnp.ravel(scores)

    def parameter_score(self, states, actions):
        return vmap(self.state_action_score_jit)(states, actions)

    def compute_score(self, states, actions):
        # First vmap over states and actions (batch)
        scores = self.parameter_score_jit(states, actions)

        # No need to reshape, as this order already keeps the structure you want
        return jnp.array(scores, dtype=jnp.float64)

    @abstractmethod
    def diff(self, state):
        pass

    @abstractmethod
    def reduce_exploration(self):
        pass

    @abstractmethod
    def get_parameters(self):
        pass