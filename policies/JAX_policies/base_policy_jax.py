"""Base Policy class implementation"""
# Libraries
from abc import ABC, abstractmethod
from jax import vmap, jacfwd, jit
import numpy as np

# Class
class BasePolicyJAX(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.dim_state = None
        self.dim_action = None
        self.tot_params = None
        self.jacobian = None
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
        return

    def _log_policy(self, parameters, state, action):
        pass

    def compute_score(self, state, action):

        if self.jacobian is None:
            self.compile_jacobian()

        scores = vmap(lambda params: (-1) * self.jacobian(params, np.ravel(state), action))(self.parameters)

        if self.multi_linear:
            scores = np.ravel(scores)

        return np.array(scores, dtype=np.float64)

    @abstractmethod
    def diff(self, state):
        pass

    @abstractmethod
    def reduce_exploration(self):
        pass
    
    @abstractmethod
    def get_parameters(self):
        pass
