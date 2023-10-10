"""
Summary: CPGPE implementation
Author: @MontenegroAlessandro
Date: 4/10/2023
"""
# todo: remove the cost list, we assume that an environment at each step
#  returns a list with all the values of the constraints
# Libraries
from algorithms.pgpe import PGPE
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import RhoElem, LearnRates
import json, io, os, errno
from tqdm import tqdm


# Algorithm implementation
class CPGPE(PGPE):
    """Class implementing CPGPE"""
    def __init__(
            self, lr=None, initial_rho: np.array = None, ite: int = 0,
            batch_size: int = 10, episodes_per_theta: int = 10,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "", verbose: bool = False, natural: bool = False,
            conf_values: list = None, constraints: list = None,
            costs: list = None
    ) -> None:
        """
        Args:
            from "lr" to "natural" see PGPE class

            conf_values (list): it is a list of confidence values needed to
            compute the CVaR_{alpha} of the costs. Each element of the list
            must be a float.

            constraints (list): it is a list of thresholds for the CVaR_{alpha}
            of the costs, Each element must be a float.

            costs (list): it is a list of cost function, used in order to
            evaluate how much each constraint is respected or not. Each element
            must be a function takin as input a state and an action, and must
            return a float.
            costs = [c_1, ..., c_K], c_i(state, reward) -> float
        """
        # Super class initialization
        super().__init__(lr=lr, initial_rho=initial_rho, ite=ite,
                         batch_size=batch_size,
                         episodes_per_theta=episodes_per_theta, env=env,
                         policy=policy, data_processor=data_processor,
                         directory=directory, verbose=verbose, natural=natural)

        # CPGPE arguments
        # Learning rates
        assert lr is not None, "[ERROR] No learning rates provided."
        assert len(lr) == 3, "[ERROR] Expected 3 learning rates."
        self.lr = lr

        # Constraints
        err_msg = "[ERROR] No confidence values for the constraints provided."
        assert len(conf_values) > 0, err_msg
        self.conf_values = conf_values
        self.n_constraints = len(conf_values)

        err_msg = "[ERROR] Number of thresholds != Number of constraints."
        assert len(constraints) == self.n_constraints, err_msg
        self.thresholds = constraints

        err_msg = "[ERROR] Number of costs != Number of constraints."
        assert len(costs) == self.n_constraints, err_msg
        self.costs = costs

        # Useful Structures
        self.costs_idx = np.zeros((self.n_constraints, ite), dtype=float)
        self.costs_idx_theta = np.zeros(
            (self.n_constraints, ite, batch_size),
            dtype=float
        )
        return

    def learn(self) -> None:
        """Learning function"""
        pass

    def collect_trajectory(self, params: np.array,
                           starting_state=None) -> float:
        pass

    def update_rho(self) -> None:
        pass

    def update_eta(self) -> None:
        pass

    def update_lambda(self) -> None:
        pass

    def update_best_rho(self, current_perf: float):
        pass

    def update_best_theta(self, current_perf: float, params: np.array) -> None:
        pass

    def save_results(self) -> None:
        pass