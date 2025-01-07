"""P-PES implementation"""
# Libraries
import copy
import errno
import io
import json
import os

import numpy as np
from tqdm import tqdm
# from adam.adam import Adam
from algorithms.utils import LearnRates, check_directory_and_create, ParamSamplerResults
# from data_processors import IdentityDataProcessor
from algorithms.samplers import *
from algorithms.pgpe import PGPE
from algorithms.policy_gradient import PolicyGradient

class PES:
    def __init__(
            self,
            phases: int = 100,
            initial_sigma: float = 1.0,
            sigma_exponent: float = 1.0,
            pg_sub_name: str = "PGPE",
            pg_sub_dict: dict = None,
            directory: str = "",
            checkpoint_freq: int = 1,
            last_rate: float = 0.1
    ) -> None:
        """
        Args:
            phases: number of phases (i.e., macro-iterations) the algorithm has to do
            initial_sigma: the multiplicative factor for the \sigma_{0} (t + 1)^{-y}
            sigma_exponent: the "y" term in the sigma update rule

            For the other parameters, please check "PGPE"
        """

        # Number of phases
        err_msg = "[PES] Error in the number of phases."
        assert phases > 0, err_msg
        self.phases = phases

        # Exploration Scheduler
        err_msg = "[PES] Invalid initial exploration."
        assert initial_sigma > 0, err_msg
        self.initial_sigma = initial_sigma

        err_msg = "[PES] Invalid exploration exponent."
        assert sigma_exponent > 0, err_msg
        self.sigma_exponent = sigma_exponent

        # Initialization of the exploration
        self.current_phase = 0
        self.sigma = 0
        self._update_sigma()

        # PGPE subroutine initialization
        err_msg = "[PES Invalid pg subroutine name."
        assert pg_sub_name in ["PGPE", "PG"], err_msg
        if pg_sub_name == "PGPE":
            self.pg_sub_class = PGPE
        else:
            self.pg_sub_class = PolicyGradient
        self.pg_sub_name = pg_sub_name
        self.pg_sub_args = copy.deepcopy(pg_sub_dict)
        self.pg_sub = None
        
        # Log
        err_msg = "[PES] Invalid checkpoint frequency."
        assert checkpoint_freq, err_msg
        self.checkpoint_freq = checkpoint_freq
        
        self.directory = directory
        if directory is not None:
            check_directory_and_create(self.directory)

        # Saving stuff
        self.sigmas = np.zeros(self.phases)
        self.sigmas[0] = self.sigma
        self.performances = np.zeros(self.phases)
        self.last_param = None
        self.last_rate = last_rate
    
    def learn(self):
        for i in tqdm(range(self.phases)):
            # Init PG Subroutine
            self._init_pg_sub()

            # Run PG subroutine
            self.pg_sub.learn()

            # Save Last Performance
            num_elem = int(self.last_rate * len(self.pg_sub.performance_idx))
            self.performances[i] = np.mean(self.pg_sub.performance_idx[-num_elem:])

            # Save Last Parameters
            self._inject_parameters()

            # Log results
            print(f"\nPhase {i}")
            print(f"Exploration {self.sigma}")
            print(f"Performance {self.performances[i]}")

            # Update Sigma
            self.current_phase += 1
            self._update_sigma()
            if(i + 1 < self.phases):
                self.sigmas[i+1] = self.sigma

            # Save results
            if (i == 0 or self.checkpoint_freq % i == 0 or i == self.phases - 1) and self.directory is not None:
                self.save_results()


    def _inject_parameters(self):
        # TODO: fare finestra
        if self.pg_sub_name == "PGPE":
            # self.pg_sub_args["initial_rho"][RhoElem.MEAN] = copy.deepcopy(self.pg_sub.rho[RhoElem.MEAN])
            self.pg_sub_args["initial_rho"][RhoElem.MEAN] = copy.deepcopy(self.pg_sub.best_rho[RhoElem.MEAN])
            self.last_param = copy.deepcopy(self.pg_sub_args["initial_rho"][RhoElem.MEAN])
        else:
            # self.pg_sub_args["initial_theta"] = copy.deepcopy(self.pg_sub.thetas)
            self.pg_sub_args["initial_theta"] = copy.deepcopy(self.pg_sub.best_theta)
            self.last_param = copy.deepcopy(self.pg_sub_args["initial_theta"])

    def _init_pg_sub(self):
        self.pg_sub = self.pg_sub_class(**self.pg_sub_args)
        self._inject_sigma()

    def _update_sigma(self) -> None:
        self.sigma = self.initial_sigma * np.power(self.current_phase + 1, -self.sigma_exponent)
    
    def _inject_sigma(self):
        if self.pg_sub_name == "PGPE":
            self.pg_sub.rho[RhoElem.STD] = self.sigma
        else:
            self.pg_sub.policy.std_dev = self.sigma
    
    def save_results(self) -> None:
        # Create the dictionary with the useful info
        results = {
            "performances": np.array(self.performances, dtype=float).tolist(),
            "sigmas": np.array(self.sigmas, dtype=float).tolist(),
            "last_param": np.array(self.last_param, dtype=float).tolist()
        }

        # Save the json
        c = "p"
        if self.pg_sub_name == "PG":
            c = "a"
        name = self.directory + f"/{c}pes_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return