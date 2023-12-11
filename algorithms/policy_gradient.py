"""
Summary: Policy Gradient Implementation
Author: @MontenegroAlessandro
Date: 6/12/2023
# todo natural
# todo baseline
"""
# Libraries
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import TrajectoryResults, check_directory_and_create
from algorithms.samplers import TrajectorySampler, pg_sampling_worker
from joblib import Parallel, delayed
import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam


# Class Implementation
class PolicyGradient:
    """This Class implements Policy Gradient Algorithms via REINFORCE or GPOMDP."""
    def __init__(
            self, lr: np.array = None,
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
            n_jobs: int = 1
    ) -> None:
        # Class' parameter with checks
        err_msg = "[PG] lr size is different wrt the one of parameters!"
        assert len(lr) == len(initial_theta), err_msg
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
        self.n_jobs = n_jobs
        self.parallel_computation = bool(self.n_jobs != 1)

        # Useful structures
        self.theta_history = np.zeros((self.ite, self.dim), dtype=np.float128)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float128)
        self.best_theta = np.zeros(self.dim, dtype=np.float128)
        self.best_performance_theta = -np.inf
        self.sampler = TrajectorySampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor
        )

        # init the theta history
        self.theta_history[self.time, :] = copy.deepcopy(self.thetas)

        # create the adam optimizers
        if self.lr_strategy == "adam":
            self.adam_optimizers = []
            for i in range(self.dim):
                self.adam_optimizers.append(Adam(self.lr[i], strategy="ascent"))
        return

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            if self.parallel_computation:
                # prepare the parameters
                worker_dict = dict(
                    env=copy.deepcopy(self.env),
                    pol=copy.deepcopy(self.policy),
                    dp=copy.deepcopy(self.data_processor),
                    params=copy.deepcopy(self.thetas),
                    starting_state=None
                )

                # build the parallel functions
                delayed_functions = delayed(pg_sampling_worker)

                # parallel computation
                res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed_functions(**worker_dict) for _ in range(self.batch_size)
                )

            else:
                res = []
                for j in range(self.batch_size):
                    tmp_res = self.sampler.collect_trajectory(params=copy.deepcopy(self.thetas))
                    res.append(tmp_res)

            # Update performance
            perf_vector = np.zeros(self.batch_size, dtype=np.float128)
            score_vector = np.zeros((self.batch_size, self.env.horizon, self.dim),
                                    dtype=np.float128)
            reward_vector = np.zeros((self.batch_size, self.env.horizon), dtype=np.float128)
            for j in range(self.batch_size):
                perf_vector[j] = res[j][TrajectoryResults.PERF]
                reward_vector[j, :] = res[j][TrajectoryResults.RewList]
                score_vector[j, :, :] = res[j][TrajectoryResults.ScoreList]
            self.performance_idx[i] = np.mean(perf_vector)

            # Update best rho
            self.update_best_theta(current_perf=self.performance_idx[i])

            # Compute the estimated gradient
            if self.estimator_type == "REINFORCE":
                estimated_gradient = np.mean(
                    perf_vector[:, np.newaxis] * np.sum(score_vector, axis=1), axis=0)
            elif self.estimator_type == "GPOMDP":
                estimated_gradient = self.update_g(
                    reward_trajectory=reward_vector, score_trajectory=score_vector
                )
            else:
                err_msg = f"[PG] {self.estimator_type} has not been implemented yet!"
                raise NotImplementedError(err_msg)

            # Update parameters
            if self.lr_strategy == "constant":
                self.thetas = self.thetas + self.lr * estimated_gradient
            elif self.lr_strategy == "adam":
                adaptive_lr = []
                for j in range(self.dim):
                    adaptive_lr.append(self.adam_optimizers[j].compute_gradient(estimated_gradient[j]))
                adaptive_lr = np.array(adaptive_lr)
                self.thetas = self.thetas + adaptive_lr
            else:
                err_msg = f"[PG] {self.lr_strategy} not implemented yet!"
                raise NotImplementedError(err_msg)

            # Log
            if self.verbose:
                print("*" * 30)
                print(f"Step: {self.time}")
                print(f"Mean Performance: {self.performance_idx[self.time - 1]}")
                print(f"Estimated gradient: {estimated_gradient}")
                print(f"Parameter (new) values: {self.thetas}")
                print(f"Best performance so far: {self.best_performance_theta}")
                print(f"Best configuration so far: {self.best_theta}")
                print("*" * 30)

            # Checkpoint
            if self.time % self.checkpoint_freq == 0:
                self.save_results()

            # save theta history
            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)

            # time update
            self.time += 1

            # reduce the exploration factor of the policy
            self.policy.reduce_exploration()
        return

    def update_g(
            self, reward_trajectory: np.array,
            score_trajectory: np.array
    ) -> np.array:
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float128)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(score_trajectory, axis=1)
        reward_trajectory = reward_trajectory[:, :, np.newaxis] * rolling_scores
        estimated_gradient = np.mean(
            np.sum(gamma_seq[:, np.newaxis] * reward_trajectory, axis=1),
            axis=0)
        return estimated_gradient

    def update_best_theta(self, current_perf: np.float128) -> None:
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
            "performance": np.array(self.performance_idx, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "thetas_history": np.array(self.theta_history, dtype=float).tolist(),
            "last_theta": np.array(self.thetas, dtype=float).tolist(),
            "best_perf": float(self.best_performance_theta)
        }

        # Save the json
        name = self.directory + "/pg_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
