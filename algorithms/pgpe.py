"""PGPE implementation"""
# Libraries
import copy
import errno
import io
import json
import os

import numpy as np
from tqdm import tqdm
from adam.adam import Adam
from algorithms.utils import LearnRates, check_directory_and_create, ParamSamplerResults
from data_processors import IdentityDataProcessor
from algorithms.samplers import *


# Objects
class PGPE:
    """Class implementing PGPE exploiting a Gaussian Hyper-Policy."""
    def __init__(
            self,
            lr: list = None,
            initial_rho: np.array = None,
            ite: int = 100,
            batch_size: int = 10,
            episodes_per_theta: int = 10,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "",
            verbose: bool = False,
            natural: bool = False,
            checkpoint_freq: int = 1,
            lr_strategy: str = "constant",
            learn_std: bool = False,
            std_decay: float = 0,
            std_min: float = 1e-4,
            n_jobs_param: int = 1,
            n_jobs_traj: int = 1,
            save_det: int = 0
    ) -> None:
        """
        Args:
            lr (float, optional): learning rate. Defaults to 1e-3.
            
            initial_rho (np.array, optional): Initial configuration of the
            hyperpolicy. Each element is assumed to be an array containing
            "[mean, log(std_dev)]". Defaults to None.
            
            ite (int, optional): Number of required iterations. Defaults to 0.
            
            batch_size (int, optional): How many theta to sample for each rho
            configuration. Defaults to 10.
            
            episodes_per_theta (int, optional): How many episodes to sample for 
            each theta configuration. Defaults to 10.
            
            env (BaseEnv, optional): The environment in which the agent has to 
            act. Defaults to None.
            
            policy (BasePolicy, optional): The parametric policy to use. 
            Defaults to None.
            
            data_processor (IdentityDataProcessor, optional): the object in 
            charge of transforming the state into a feature vector. Defaults to 
            None.
            
            directory (str, optional): where to save the results
            
            natural (bool): whether to use the natural gradient
            
            verbose (bool): whether to log additional information. 
            Defaults to FALSE.
            
            checkpoint_freq (int): after how many iterations results are saved.
            Defualts to 1.
            
            lr_strategy (str): how to update the learning rate. Choises in 
            "constant" or "adam". Defaults to "adam".
            
            learn_std (bool): whether to learn the standard deviation of the 
            hyper-policy. Defaults to False.
            
            std_decay (float): how much to decrease the standard deviation at 
            each iteration of the algorithm. Defaults to 0 (i.e., no decay).
            
            std_min (float): the minimum value the standard deviation can 
            assume. Defaults to 1e-4.
            
            n_jobs_param (int): how many parameters sampled are tested in 
            parallel. Defaults to 1.
            
            n_jobs_traj (int): how many trajectories (for each parameter 
            sampled) are evaluated in parallel. Defaults to 1.
        """
        # Arguments with checks
        assert lr is not None, "[ERROR] No Learning rate provided"
        self.lr = lr[LearnRates.PARAM]

        assert initial_rho is not None, "[ERROR] No initial hyperpolicy."
        self.rho = np.array(initial_rho, dtype=np.float64)
        self.dim = len(self.rho[RhoElem.MEAN])

        assert env is not None, "[ERROR] No env provided."
        self.env = env

        assert policy is not None, "[ERROR] No policy provided."
        self.policy = policy

        assert data_processor is not None, "[ERROR] No data processor."
        self.data_processor = data_processor

        self.directory = directory
        if self.directory is not None:
            check_directory_and_create(self.directory)
        self.save_det = save_det

        err_msg = "[PGPE] The lr_strategy is not valid."
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy
        if self.lr_strategy == "adam":
            self.rho_adam = [None, None]
            self.rho_adam[RhoElem.MEAN] = Adam(step_size=self.lr, strategy="ascent")
            self.rho_adam[RhoElem.STD] = Adam(step_size=self.lr, strategy="ascent")

        # Arguments without check
        self.ite = ite
        self.batch_size = batch_size
        self.episodes_per_theta = episodes_per_theta
        self.verbose = verbose
        self.natural = natural
        self.learn_std = learn_std
        self.std_decay = std_decay
        self.std_min = std_min
        self.n_jobs_param = n_jobs_param
        self.n_jobs_traj = n_jobs_traj

        # Additional parameters
        if len(self.rho[RhoElem.STD]) != self.dim:
            raise ValueError("[PGPE] different size in RHO for µ and σ.")
        self.thetas = np.zeros((self.batch_size, self.dim), dtype=np.float64)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.performance_idx_theta = np.zeros((ite, batch_size), dtype=np.float64)
        self.parallel_computation_param = bool(self.n_jobs_param != 1)
        self.parallel_computation_traj = bool(self.n_jobs_traj != 1)
        self.sampler = ParameterSampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor,
            episodes_per_theta=self.episodes_per_theta, n_jobs=self.n_jobs_traj
        )

        # Saving parameters
        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_rho = self.rho
        self.best_performance_theta = -np.inf
        self.best_performance_rho = -np.inf
        self.checkpoint_freq = checkpoint_freq
        self.deterministic_curve = np.zeros(self.ite)

        self.rho_history = np.zeros((ite, self.dim), dtype=np.float64)
        self.rho_history[0, :] = copy.deepcopy(self.rho[RhoElem.MEAN])

        # Varaince learning
        self.std_score = None

        return

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            # Collect the results
            if self.parallel_computation_param:
                worker_dict = dict(
                    env=copy.deepcopy(self.env),
                    pol=copy.deepcopy(self.policy),
                    dp=copy.deepcopy(self.data_processor),
                    params=copy.deepcopy(self.rho),
                    episodes_per_theta=self.episodes_per_theta,
                    n_jobs=self.n_jobs_traj
                )
                delayed_functions = delayed(pgpe_sampling_worker)
                res = Parallel(n_jobs=self.n_jobs_param)(
                    delayed_functions(**worker_dict) for _ in range(self.batch_size)
                )
            else:
                res = []
                for j in range(self.batch_size):
                    res.append(self.sampler.collect_trajectories(params=copy.deepcopy(self.rho)))

            # post-processing of results
            performance_res = np.zeros(self.batch_size, dtype=np.float64)
            for z in range(self.batch_size):
                self.thetas[z, :] = res[z][ParamSamplerResults.THETA]
                performance_res[z] = np.mean(res[z][ParamSamplerResults.PERF])
            self.performance_idx_theta[i, :] = performance_res

            # try to update the best theta
            max_batch_perf = np.max(performance_res)
            best_theta_batch_index = np.where(performance_res == max_batch_perf)[0]
            self.update_best_theta(
                current_perf=max_batch_perf, params=self.thetas[best_theta_batch_index, :]
            )

            # Update performance
            self.performance_idx[i] = np.mean(self.performance_idx_theta[i, :])

            # Update best rho
            self.update_best_rho(current_perf=self.performance_idx[i])

            # Update parameters
            self.update_rho()

            # save the current rho configuration
            self.rho_history[self.time, :] = copy.deepcopy(self.rho[RhoElem.MEAN])

            # Update time counter
            self.time += 1
            if self.verbose:
                print(f"rho perf: {self.performance_idx}")
                print(f"theta perf: {self.performance_idx_theta}")
            if self.time % self.checkpoint_freq == 0 and self.directory is not None:
                self.save_results()

            # std_decay
            if not self.learn_std:
                # std = np.float64(np.exp(self.rho[RhoElem.STD]))
                std = self.rho[RhoElem.STD]
                std = np.clip(std - self.std_decay, self.std_min, np.inf)
                self.rho[RhoElem.STD, :] = std # np.log(std)

        # Sample the deterministic curve
        if self.save_det:
            self.sample_deterministic_curve()

        return

    def update_rho(self) -> None:
        """This function modifies the self.rho vector, by updating via the 
        estimated gradient."""
        # Take the performance of the whole batch
        batch_perf = self.performance_idx_theta[self.time, :]

        # take the means and the sigmas
        means = self.rho[RhoElem.MEAN, :]
        # stds = np.float64(np.exp(self.rho[RhoElem.STD, :]))
        stds = self.rho[RhoElem.STD, :]

        # compute the scores
        if not self.natural:
            log_nu_means = (self.thetas - means) / (stds ** 2)
            log_nu_stds = (((self.thetas - means) ** 2) - (stds ** 2)) / (stds ** 2)
        else:
            log_nu_means = self.thetas - means
            log_nu_stds = (((self.thetas - means) ** 2) - (stds ** 2)) / (2 * (stds ** 2))

        # compute the gradients
        grad_means = batch_perf[:, np.newaxis] * log_nu_means
        grad_stds = batch_perf[:, np.newaxis] * log_nu_stds

        # update rho
        if self.lr_strategy == "constant":
            self.rho[RhoElem.MEAN, :] = self.rho[RhoElem.MEAN, :] + self.lr * np.mean(grad_means, axis=0)
            # update sigma if it is the case
            # if self.learn_std:
            #     self.rho[RhoElem.STD, :] = self.rho[RhoElem.STD, :] + self.lr * np.mean(grad_stds, axis=0)
        elif self.lr_strategy == "adam":
            adaptive_lr_m = self.rho_adam[RhoElem.MEAN].compute_gradient(np.mean(grad_means, axis=0))
            adaptive_lr_m = np.array(adaptive_lr_m)
            self.rho[RhoElem.MEAN, :] = self.rho[RhoElem.MEAN, :] + adaptive_lr_m
            # update sigma if it is the case
            # if self.learn_std:
            #     adaptive_lr_s = self.rho_adam[RhoElem.STD].compute_gradient(np.mean(grad_stds, axis=0))
            #     adaptive_lr_s = np.array(adaptive_lr_s)
            #     self.rho[RhoElem.STD, :] = self.rho[RhoElem.STD, :] + adaptive_lr_s
        else:
            raise NotImplementedError("[PGPE] Ops, not implemented yet!")
        
        # compute sigma score if needed
        if self.learn_std:
            std = stds[0] 
            self.std_score = (((np.linalg.norm(self.thetas - means, axis=1) ** 2) - self.dim * (std ** 2)) / (std ** 3))
            self.std_score = self.std_score * batch_perf
        return

    def sample_theta(self, index: int) -> None:
        """
        Summary:
            This function modifies the self.thetas vector, by sampling parameters
            from the current rho configuration. Rho is assumed to be of the form: 
            "[[means...], [log(std_devs)...]]".
        Args:
            index (int): the current index of the thetas vector
        """
        for id in range(len(self.rho[RhoElem.MEAN])):
            # self.thetas[index, id] = np.random.normal(
            #     loc=self.rho[RhoElem.MEAN, id],
            #     scale=np.exp(np.float64(self.rho[RhoElem.STD, id]))
            # )
            self.thetas[index, id] = np.random.normal(
                loc=self.rho[RhoElem.MEAN, id],
                scale=self.rho[RhoElem.STD, id]
            )
        return

    def sample_theta_from_best(self):
        """
        Summary:
            This function samples a parameter configuration $\\theta$ from 
            the best hyperparameter configuration found so far.

        Returns:
            np.array: the sampled parameter vector.
        """
        thetas = []
        for id in range(len(self.best_rho[RhoElem.MEAN])):
            # thetas.append(np.random.normal(
            #     loc=self.rho[RhoElem.MEAN, id],
            #     scale=np.exp(np.float64(self.rho[RhoElem.STD, id])))
            # )
            thetas.append(np.random.normal(
                loc=self.rho[RhoElem.MEAN, id],
                scale=self.rho[RhoElem.STD, id])
            )
        return thetas

    def collect_trajectory(self, params: np.array, starting_state=None) -> float:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta 
            configuration.
        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): teh starting state for the iterations
        Returns:
            float: the discounted reward of the trajectory
        """
        # reset the environment
        self.env.reset()
        if starting_state is not None:
            self.env.state = copy.deepcopy(starting_state)

        # initialize parameters
        perf = 0
        self.policy.set_parameters(thetas=params)

        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state

            # transform the state
            features = self.data_processor.transform(state=state)

            # select the action
            a = self.policy.draw_action(state=features)

            # play the action
            _, rew, abs = self.env.step(action=a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew

            if self.verbose:
                print("*" * 30)
                print(f"ACTION: {a}")
                print(f"FEATURES: {features}")
                print(f"REWARD: {rew}")
                print(f"PERFORMANCE: {perf}")
                print("*" * 30)
            
            if abs:
                break

        return perf

    def update_best_rho(self, current_perf: float, *args, **kwargs) -> None:
        """
        Summary:
            Function updating the best configuration found so far
        Args:
            current_perf (float): current performance value to evaluate
        """
        if current_perf > self.best_performance_rho:
            self.best_rho = self.rho
            self.best_performance_rho = current_perf
            print("-" * 30)
            print(f"New best RHO: {self.best_rho}")
            print(f"New best PERFORMANCE: {self.best_performance_rho}")
            print("-" * 30)

            # Save the best rho configuration
            if self.directory is not None:
                if self.directory != "":
                    file_name = self.directory + "/best_rho"
                else:
                    file_name = "best_rho"
                np.save(file_name, self.best_rho)
        return

    def update_best_theta(self, current_perf: float, params: np.array, *args, **kwargs) -> None:
        """
        Summary:
            Function updating the best configuration found so far
        Args:
            current_perf (float): current performance value to evaluate
            params (np.array): the current sampling of theta values
        """
        if current_perf > self.best_performance_theta:
            self.best_theta = params
            self.best_performance_theta = current_perf
            """print("*" * 30)
            print(f"New best THETA: {self.best_theta}")
            print(f"New best PERFORMANCE: {self.best_performance_theta}")
            print("*" * 30)"""

            # Save the best theta configuration
            if self.directory is not None:
                if self.directory != "":
                    file_name = self.directory + "/best_theta"
                    
                else:
                    file_name = "best_theta"
                np.save(file_name, self.best_theta)
        return

    def sample_deterministic_curve(self):
        """
        Summary:
            This sample computes the deterministic curve associated with the 
            sequence of hyperparameter configuration seen during the learning.
        """
        for i in tqdm(range(self.ite)):
            self.policy.set_parameters(thetas=self.rho_history[i, :])
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=IdentityDataProcessor(),
                # params=copy.deepcopy(self.rho_history[i, :]),
                params=None,
                starting_state=None
            )
            # build the parallel functions
            delayed_functions = delayed(pg_sampling_worker)

            # parallel computation
            res = Parallel(n_jobs=self.n_jobs_param, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(self.batch_size)
            )

            # extract data
            ite_perf = np.zeros(self.batch_size, dtype=np.float64)
            for j in range(self.batch_size):
                ite_perf[j] = res[j][TrajectoryResults.PERF]

            # compute mean
            self.deterministic_curve[i] = np.mean(ite_perf)

    def save_results(self) -> None:
        """Function saving the results of the training procedure"""
        # Create the dictionary with the useful info
        results = {
            "performance_rho": np.array(self.performance_idx, dtype=float).tolist(),
            "performance_thetas_per_rho": np.array(self.performance_idx_theta, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "best_rho": np.array(self.best_rho, dtype=float).tolist(),
            "thetas_history": np.array(self.thetas, dtype=float).tolist(),
            "rho_history": np.array(self.rho_history, dtype=float).tolist(),
            "deterministic_res": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/pgpe_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
