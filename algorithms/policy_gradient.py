"""
Summary: Policy Gradient Implementation
Author: @MontenegroAlessandro
Date: 6/12/2023
"""
# Libraries
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import TrajectoryResults, LearnRates, check_directory_and_create
from algorithms.trajectory_sampler import PGTrajectorySampler
from joblib import Parallel, delayed
import json, io, os, errno
from tqdm import tqdm
import copy
from adam.adam import Adam


# Class Implementation
class PolicyGradient:
    """This Class implements Policy Gradient Algorithms via REINFORCE or GPOMDP."""
    def __init__(
            self, lr: float = 1e-3,
            lr_strategy: str = "constant",
            estimator_type: str = "REINFORCE",
            initial_theta: np.array = None,
            ite: int = 100,
            batch_size: int = 1,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "",
            verbose: bool = False,
            natural: bool = False,
            checkpoint_freq: int = 1,
            parallel_computation: bool = False
    ) -> None:
        # Class' parameter with checks
        err_msg = "[PG] lr value cannot be negative!"
        assert lr > 0, err_msg
        self.lr = lr

        err_msg = "[PG] lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy

        err_msg = "[PG] estimator_type not valid!"
        assert estimator_type in ["REINFORCE", "GPOMDP"], err_msg
        self.estimator_type = estimator_type

        err_msg = "[PG] initial_theta has not been specified!"
        assert initial_theta is not None, err_msg
        self.thetas = np.array(initial_theta)
        self.dim = len(self.thetas)

        err_msg = "[PG] env is None."
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PG] policy is None."
        assert policy is not None, err_msg
        self.policy = policy

        err_msg = "[PG] data processor is None."
        assert data_processor is not None, err_msg
        self.data_processor = data_processor

        check_directory_and_create(dir_name=directory)
        self.directory = directory

        # Other class' parameters
        self.ite = ite
        self.batch_size = batch_size
        self.verbose = verbose
        self.natural = natural
        self.checkpoint_freq = checkpoint_freq
        self.parallel_computation = parallel_computation

        # Useful structures
        self.theta_history = np.zeros((self.ite, self.dim), dtype=float)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=float)
        self.best_theta = np.zeros(self.dim, dtype=float)
        self.best_performance_theta = -np.inf
        self.sampler = PGTrajectorySampler(env=self.env,
                                           pol=self.policy,
                                           data_processor=self.data_processor)

        # init the theta history
        self.theta_history[self.time, :] = copy.deepcopy(self.thetas)
        return

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            if self.parallel_computation:
                delayed_functions = delayed(self.sampler.collect_trajectory)(copy.deepcopy(self.thetas))
                r = Parallel(n_jobs=self.batch_size, backend="loky")(delayed_functions)
                res, _ = zip(*r)
            else:
                res = []
                for j in range(self.batch_size):
                    perf, rew_list, score_list = self.sampler.collect_trajectory(params=copy.deepcopy(self.thetas))
                    res.append(self.sampler.collect_trajectory(params=copy.deepcopy(self.thetas)))

            # Update performance
            self.performance_idx[i] = np.mean(res)

            # Update best rho
            self.update_best_theta(current_perf=self.performance_idx[i])

            # Update parameters
            if self.estimator_type == "REINFORCE":
                self.update_r()
            elif self.estimator_type == "GPOMDP":
                self.update_g()
            else:
                raise NotImplementedError(f"{self.estimator_type} has not been implemented yet!")

            # Update time counter
            self.time += 1
            if self.verbose:
                pass
            if self.time % self.checkpoint_freq == 0:
                self.save_results()

            # save theta history
            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)
        return

    def update_r(self) -> None:
        pass

    def update_g(self) -> None:
        pass

    def update_best_theta(self, current_perf: float) -> None:
        if self.best_theta is None or self.best_performance_theta <= current_perf:
            self.best_performance_theta = current_perf
            self.best_theta = copy.deepcopy(self.thetas)

            if self.verbose:
                print("#" * 30)
                print("New best parameter configuration found")
                print(f"Performance: {self.best_performance_theta}")
                print(f"Parameter configuration: {self.best_theta}")
                print("#" * 30)
        return

    def save_results(self) -> None:
        results = {
            "performance": self.performance_idx.tolist(),
            "best_theta": self.best_theta.tolist(),
            "thetas_history": self.theta_history.tolist(),
            "last_theta": self.thetas.tolist(),
            "best_perf": self.best_performance_theta
        }

        # Save the json
        name = self.directory + "/pg_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
