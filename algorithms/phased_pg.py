"""P-PES implementation"""
# Libraries
import copy
import errno
import io
import json
import os

import numpy as np
from tqdm import tqdm
from adam.adam import Adam
from algorithms.utils import LearnRates, check_directory_and_create, ParamSamplerResults, TrajectoryResults2
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
            last_rate: float = 0.1,
            dim: int = 1
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

        # Dimension
        err_msg = "[PES] Invalid dimension."
        assert dim > 0, err_msg
        self.dim = dim

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
        self.sigmas = np.zeros((self.phases, dim))
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
            # TODO: fix this later
            # self.pg_sub.rho[RhoElem.STD] = self.sigma
            self.pg_sub.rho[RhoElem.STD] = np.log(self.sigma)
        else:
            if self.current_phase == 0:
                self.pg_sub.policy.std_dev = self.sigma * np.ones(self.dim)
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
    
class PEL(PES):
    """PEL Implementation, though for Gaussian (hyper)policies"""
    def __init__(
            self, 
            lr_sigma: float = 0.01,
            lr_sigma_strategy: str = "adam",
            phases = 100, 
            initial_sigma = 1, 
            sigma_exponent = 1, 
            pg_sub_name = "PGPE", 
            pg_sub_dict = None, 
            directory = "", 
            checkpoint_freq = 1, 
            last_rate = 0.1,
            dim: int = 1
    ):
        super().__init__(
            phases, 
            initial_sigma, 
            sigma_exponent, 
            pg_sub_name, 
            pg_sub_dict, 
            directory, 
            checkpoint_freq, 
            last_rate,
            dim
        )

        # Learning rate strategy
        err_msg = "[PEL] Invalid strategy for learning rate."
        assert lr_sigma_strategy in ["constant", "adam"], err_msg
        self.lr_sigma_strategy = lr_sigma_strategy
        
        # Learning rate value
        err_msg = "[PEL] Invalid learning rate value."
        assert lr_sigma > 0, err_msg
        self.lr_sigma = lr_sigma

        # Learning rate scheduler
        self.lr_scheduler = None
        if self.lr_sigma_strategy == "adam":
            self.lr_scheduler = Adam(step_size=self.lr_sigma, strategy="ascent")
        
        # Sigma initialization
        self.sigma = 0
        self.nu = np.clip(self.initial_sigma, -np.inf, 0)
        self._map_sigma()
        self.sigmas[0] = self.sigma
    
    def learn(self):
        for i in tqdm(range(self.phases)):
            # Init PG Subroutine
            self._init_pg_sub()

            # Run PG subroutine
            self.pg_sub.learn()

            # Save Last Performance
            num_elem = int(self.last_rate * len(self.pg_sub.performance_idx))
            self.performances[i] = np.mean(self.pg_sub.performance_idx[-num_elem:])

            # Update Exploration Parameterization
            self._extract_parameters()
            if self.pg_sub_name == "PGPE":
                self._parameter_based_gradient()
            else:
                self._action_based_gradient()
            self.nu = np.clip(self.nu, -np.inf, 0)

            # Save Last Parameters
            self._inject_parameters()

            # Log results
            print(f"\nPhase {i}")
            print(f"Exploration {self.sigma}")
            print(f"Performance {self.performances[i]}")

            # Update Sigma
            self.current_phase += 1
            self._map_sigma()
            if(i + 1 < self.phases):
                self.sigmas[i+1] = self.sigma

            # Save results
            if (i == 0 or self.checkpoint_freq % i == 0 or i == self.phases - 1) and self.directory is not None:
                self.save_results()

    def _parameter_based_gradient(self):
        # Array to store the parameters seen during the trajectories
        thetas = np.zeros((self.pg_sub.batch_size, self.pg_sub.dim))
        performance_res = np.zeros(self.pg_sub.batch_size, dtype=np.float64)

        # Collect the results
        self.pg_sub.rho[RhoElem.MEAN] = copy.deepcopy(self.last_param)
        if self.pg_sub.parallel_computation_param:
            worker_dict = dict(
                env=copy.deepcopy(self.pg_sub.env),
                pol=copy.deepcopy(self.pg_sub.policy),
                dp=copy.deepcopy(self.pg_sub.data_processor),
                params=copy.deepcopy(self.pg_sub.rho),
                episodes_per_theta=self.pg_sub.episodes_per_theta,
                n_jobs=self.pg_sub.n_jobs_traj
            )
            delayed_functions = delayed(pgpe_sampling_worker)
            res = Parallel(n_jobs=self.pg_sub.n_jobs_param)(
                delayed_functions(**worker_dict) for _ in range(self.pg_sub.batch_size)
            )
        else:
            res = []
            for j in range(self.pg_sub.batch_size):
                res.append(self.pg_sub.sampler.collect_trajectories(params=copy.deepcopy(self.pg_sub.rho)))

        # post-processing of results
        for z in range(self.pg_sub.batch_size):
            thetas[z, :] = res[z][ParamSamplerResults.THETA]
            performance_res[z] = np.mean(res[z][ParamSamplerResults.PERF])
        
        # Take the performance of the whole batch
        batch_perf = performance_res

        # take the means and the sigmas
        means = self.last_param
        stds = np.ones(self.pg_sub.dim, dtype=np.float64) * self.sigma

        # compute the scores
        log_nu_stds = ((((thetas - means) ** 2) - (stds ** 2)) / (stds ** 3)) * self._grad_nu()

        # compute the gradients
        grad_stds = batch_perf[:, np.newaxis] * log_nu_stds

        # update nu
        if self.lr_sigma_strategy == "constant":
            self.nu = self.nu + self.lr_sigma * np.mean(grad_stds, axis=0)
        elif self.lr_sigma_strategy == "adam":
            adaptive_lr_s = self.lr_scheduler.compute_gradient(np.mean(grad_stds, axis=0))
            adaptive_lr_s = np.array(adaptive_lr_s)
            self.nu = self.nu + adaptive_lr_s

    def _action_based_gradient(self):
        if self.pg_sub.parallel_computation:
            # prepare the parameters
            worker_dict = dict(
                env=copy.deepcopy(self.pg_sub.env),
                pol=copy.deepcopy(self.pg_sub.policy),
                dp=copy.deepcopy(self.pg_sub.data_processor),
                params=copy.deepcopy(self.last_param),
                learn_std=True,
                e_parameterization_score=copy.deepcopy(self._grad_nu())
            )

            # build the parallel functions
            delayed_functions = delayed(pg_sampling_worker)

            # parallel computation
            res = Parallel(n_jobs=self.pg_sub.n_jobs, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(self.pg_sub.batch_size)
            )

        else:
            res = []
            for j in range(self.pg_sub.batch_size):
                tmp_res = self.sampler.collect_trajectory(params=copy.deepcopy(self.last_param))
                res.append(tmp_res)

        # Update performance
        perf_vector = np.zeros(self.pg_sub.batch_size, dtype=np.float64)
        # score_vector = np.zeros((self.pg_sub.batch_size, self.pg_sub.env.horizon, self.pg_sub.dim), dtype=np.float64)
        reward_vector = np.zeros((self.pg_sub.batch_size, self.pg_sub.env.horizon), dtype=np.float64)
        e_score_vector = np.zeros((self.pg_sub.batch_size, self.pg_sub.env.horizon, self.pg_sub.dim_action),
                                dtype=np.float64)
        for j in range(self.pg_sub.batch_size):
            perf_vector[j] = res[j][TrajectoryResults2.PERF]
            reward_vector[j, :] = res[j][TrajectoryResults2.RewList]
            # score_vector[j, :, :] = res[j][TrajectoryResults2.ScoreList]
            e_score_vector[j, :, :] = res[j][TrajectoryResults2.Info]["e_scores"]

        # GPOMDP-like estimator
        gamma = self.pg_sub.env.gamma
        horizon = self.pg_sub.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(e_score_vector, axis=1)
        reward_trajectory = reward_vector[:, :, np.newaxis] * rolling_scores
        estimated_gradient = np.mean(
            np.sum(gamma_seq[:, np.newaxis] * reward_trajectory, axis=1),
            axis=0)
        
        # Update nu
        if self.lr_sigma_strategy == "constant":
            self.nu = self.nu + self.lr_sigma * estimated_gradient
        elif self.lr_sigma_strategy == "adam":
            adaptive_lr_s = self.lr_scheduler.compute_gradient(estimated_gradient)
            adaptive_lr_s = np.array(adaptive_lr_s)
            self.nu = self.nu + adaptive_lr_s

    def _extract_parameters(self):
        # TODO: fare finestra
        if self.pg_sub_name == "PGPE":
            # self.pg_sub_args["initial_rho"][RhoElem.MEAN] = copy.deepcopy(self.pg_sub.rho[RhoElem.MEAN])
            self.pg_sub_args["initial_rho"][RhoElem.MEAN] = copy.deepcopy(self.pg_sub.best_rho[RhoElem.MEAN])
            self.last_param = copy.deepcopy(self.pg_sub_args["initial_rho"][RhoElem.MEAN])
        else:
            # self.pg_sub_args["initial_theta"] = copy.deepcopy(self.pg_sub.thetas)
            self.pg_sub_args["initial_theta"] = copy.deepcopy(self.pg_sub.best_theta)
            self.last_param = copy.deepcopy(self.pg_sub_args["initial_theta"])
    
    def _map_sigma(self):
        self.sigma = np.power(1 - self.nu, -self.sigma_exponent)
    
    def _grad_nu(self):
        return self.sigma_exponent * np.power(1 - self.nu, -self.sigma_exponent - 1)

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
        name = self.directory + f"/{c}pel_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return