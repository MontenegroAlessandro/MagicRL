"""PGPE finite-difference implementation"""
# Libraries
import copy
import io
import json

import numpy as np
from tqdm import tqdm
from adam.adam import Adam
from algorithms.utils import LearnRates, check_directory_and_create, RhoElem, TrajectoryResults
from data_processors import IdentityDataProcessor
from algorithms.samplers import *


def pgpe_fd_param_worker(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        dp: BaseProcessor = None,
        theta: np.ndarray = None,
        fd_eps: np.ndarray = None,
        param_idx: int = None,
        batch_size: int = 1,
        n_jobs_traj: int = 1,
        starting_state: np.ndarray = None,
        seed: int = 0
) -> list:
    """Worker evaluating one finite-difference perturbation over a batch."""
    # Perturb the single parameter
    perturbed_theta = copy.deepcopy(theta)
    perturbed_theta[param_idx] += fd_eps[param_idx]

    if n_jobs_traj == 1:
        sampler = TrajectorySampler(env=env, pol=pol, data_processor=dp)
        raw_res = []
        for j in range(batch_size):
            raw_res.append(sampler.collect_trajectory(params=copy.deepcopy(perturbed_theta), seed=seed+j, starting_state=starting_state))
    else:
        worker_dict = dict(
            env=copy.deepcopy(env),
            pol=copy.deepcopy(pol),
            dp=copy.deepcopy(dp),
            params=copy.deepcopy(perturbed_theta),
            starting_state=starting_state
        )
        delayed_functions = delayed(pg_sampling_worker)
        raw_res = Parallel(n_jobs=n_jobs_traj, backend="loky")(
            delayed_functions(**worker_dict, seed=seed+j) for j in range(batch_size)
        )

    perf_res = np.zeros(batch_size, dtype=np.float64)
    for i, elem in enumerate(raw_res):
        perf_res[i] = elem[TrajectoryResults.PERF]

    return [param_idx, perturbed_theta, perf_res]


# Objects
class PGPE:
    """Finite-difference policy optimizer with PGPE-compatible signature."""
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
            save_det: int = 0,
            seed: int = 0,
            starting_state: np.ndarray = None,
            fd_mode: str = "forward"
    ) -> None:
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

        err_msg = "[PGPE-FD] The lr_strategy is not valid."
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy
        self.theta_adam = None
        if self.lr_strategy == "adam":
            self.theta_adam = Adam(step_size=self.lr, strategy="ascent")

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

        if len(self.rho[RhoElem.STD]) != self.dim:
            raise ValueError("[PGPE-FD] different size in RHO for theta and epsilon.")

        self.theta = np.array(self.rho[RhoElem.MEAN], dtype=np.float64)
        self.fd_eps = np.clip(np.abs(np.array(self.rho[RhoElem.STD], dtype=np.float64)), self.std_min, np.inf)
        self.rho[RhoElem.MEAN, :] = copy.deepcopy(self.theta)
        self.rho[RhoElem.STD, :] = copy.deepcopy(self.fd_eps)

        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.performance_idx_theta = np.zeros((ite, batch_size), dtype=np.float64)
        self.parallel_computation_param = bool(self.n_jobs_param != 1)
        self.parallel_computation_traj = bool(self.n_jobs_traj != 1)

        self.sampler = TrajectorySampler(
            env=self.env,
            pol=self.policy,
            data_processor=self.data_processor
        )

        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_rho = copy.deepcopy(self.rho)
        self.best_performance_theta = -np.inf
        self.best_performance_rho = -np.inf
        self.checkpoint_freq = checkpoint_freq
        self.deterministic_curve = np.zeros(self.ite)

        self.theta_history = np.zeros((ite, self.dim), dtype=np.float64)
        self.rho_history = np.zeros((ite, self.dim), dtype=np.float64)
        self.fd_grad_history = np.zeros((ite, self.dim), dtype=np.float64)

        if self.ite > 0:
            self.theta_history[0, :] = copy.deepcopy(self.theta)
            self.rho_history[0, :] = copy.deepcopy(self.theta)

        self.std_score = None
        self.seed = seed
        self.starting_state = starting_state
        
        err_msg = "[PGPE-FD] Invalid fd_mode. Use one of: forward, central, five_point."
        assert fd_mode in ["forward", "central", "five_point"], err_msg
        self.fd_mode = fd_mode

    def collect_performance_batch(self, params, seed: np.ndarray) -> np.ndarray:
        """Collect a batch of trajectory performances for a fixed policy parameter."""
        if self.parallel_computation_traj:
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=copy.deepcopy(self.data_processor),
                params=copy.deepcopy(params),
            )
            delayed_functions = delayed(pg_sampling_worker)
            raw_res = Parallel(n_jobs=self.n_jobs_traj, backend="loky")(
                delayed_functions(**worker_dict, seed=seed+j, starting_state=self.starting_state) for j in range(self.batch_size)
            )
        else:
            raw_res = []
            for j in range(self.batch_size):
                raw_res.append(self.sampler.collect_trajectory(params=copy.deepcopy(params), seed=seed+j, starting_state=self.starting_state))

        perf_res = np.zeros(self.batch_size, dtype=np.float64)
        for i, elem in enumerate(raw_res):
            perf_res[i] = elem[TrajectoryResults.PERF]

        return perf_res

    def estimate_fd_gradient(self, base_mean_perf: float, seed: int) -> np.ndarray:
        """Estimate gradient with the selected finite-difference scheme."""
        if self.fd_mode == "forward":
            return self.estimate_fd_gradient_forward(base_mean_perf=base_mean_perf, seed=seed)
        if self.fd_mode == "central":
            return self.estimate_fd_gradient_central(seed=seed)
        if self.fd_mode == "five_point":
            return self.estimate_fd_gradient_five_point(seed=seed)
        raise NotImplementedError("[PGPE-FD] Unsupported finite-difference mode.")

    def estimate_fd_gradient_forward(self, base_mean_perf: float, seed: int) -> np.ndarray:
        """Estimate gradient with forward finite differences."""
        gradient = np.zeros(self.dim, dtype=np.float64)

        if self.parallel_computation_param:
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=copy.deepcopy(self.data_processor),
                theta=copy.deepcopy(self.theta),
                fd_eps=copy.deepcopy(self.fd_eps),
                batch_size=self.batch_size,
                n_jobs_traj=self.n_jobs_traj
            )
            delayed_functions = delayed(pgpe_fd_param_worker)
            res = Parallel(n_jobs=self.n_jobs_param, backend="loky")(
                delayed_functions(param_idx=j, seed=seed, starting_state=self.starting_state, **worker_dict) for j in range(self.dim)
            )

            for elem in res:
                j = elem[0]
                pert_theta = elem[1]
                pert_perf = elem[2]
                pert_mean = np.mean(pert_perf)
                gradient[j] = (pert_mean - base_mean_perf) / self.fd_eps[j]
                self.update_best_theta(current_perf=pert_mean, params=pert_theta)
            
        else:
            for j in range(self.dim):
                pert_theta = copy.deepcopy(self.theta)
                pert_theta[j] += self.fd_eps[j]
                pert_perf = self.collect_performance_batch(params=pert_theta, seed=seed)
                pert_mean = np.mean(pert_perf)
                gradient[j] = (pert_mean - base_mean_perf) / self.fd_eps[j]
                self.update_best_theta(current_perf=pert_mean, params=pert_theta)

        return gradient

    def estimate_fd_gradient_central(self, seed: int) -> np.ndarray:
        """Estimate gradient with central finite differences.

        dJ/dtheta_i ≈ (J(theta + delta*e_i) - J(theta - delta*e_i)) / (2*delta)
        """
        gradient = np.zeros(self.dim, dtype=np.float64)

        for j in range(self.dim):
            delta = self.fd_eps[j]

            theta_plus = copy.deepcopy(self.theta)
            theta_minus = copy.deepcopy(self.theta)
            theta_plus[j] += delta
            theta_minus[j] -= delta

            j_plus = np.mean(self.collect_performance_batch(params=theta_plus, seed=seed))
            j_minus = np.mean(self.collect_performance_batch(params=theta_minus, seed=seed))

            gradient[j] = (j_plus - j_minus) / (2.0 * delta)

            self.update_best_theta(current_perf=j_plus, params=theta_plus)
            self.update_best_theta(current_perf=j_minus, params=theta_minus)

        return gradient

    def estimate_fd_gradient_five_point(self, seed: int) -> np.ndarray:
        """Estimate gradient with 5-point centered stencil.

        dJ/dtheta_i ≈ (-J(theta+2d*e_i) + 8J(theta+d*e_i)
                       -8J(theta-d*e_i) + J(theta-2d*e_i)) / (12*d)
        """
        gradient = np.zeros(self.dim, dtype=np.float64)

        for j in range(self.dim):
            delta = self.fd_eps[j]

            theta_p2 = copy.deepcopy(self.theta)
            theta_p1 = copy.deepcopy(self.theta)
            theta_m1 = copy.deepcopy(self.theta)
            theta_m2 = copy.deepcopy(self.theta)

            theta_p2[j] += 2.0 * delta
            theta_p1[j] += delta
            theta_m1[j] -= delta
            theta_m2[j] -= 2.0 * delta

            j_p2 = np.mean(self.collect_performance_batch(params=theta_p2, seed=seed))
            j_p1 = np.mean(self.collect_performance_batch(params=theta_p1, seed=seed))
            j_m1 = np.mean(self.collect_performance_batch(params=theta_m1, seed=seed))
            j_m2 = np.mean(self.collect_performance_batch(params=theta_m2, seed=seed))

            gradient[j] = (-j_p2 + 8.0 * j_p1 - 8.0 * j_m1 + j_m2) / (12.0 * delta)

            self.update_best_theta(current_perf=j_p2, params=theta_p2)
            self.update_best_theta(current_perf=j_p1, params=theta_p1)
            self.update_best_theta(current_perf=j_m1, params=theta_m1)
            self.update_best_theta(current_perf=j_m2, params=theta_m2)

        return gradient

    def update_rho(self, estimated_gradient: np.ndarray) -> None:
        """Update policy parameters using the estimated finite-difference gradient."""
        if self.lr_strategy == "constant":
            self.theta = self.theta + self.lr * estimated_gradient
        elif self.lr_strategy == "adam":
            adaptive_lr = np.array(self.theta_adam.compute_gradient(estimated_gradient), dtype=np.float64)
            self.theta = self.theta + adaptive_lr
        else:
            raise NotImplementedError("[PGPE-FD] Ops, not implemented yet!")

        self.rho[RhoElem.MEAN, :] = copy.deepcopy(self.theta)

    def learn(self) -> None:
        """Learning function."""
        for i in tqdm(range(self.ite)):
            base_perf_batch = self.collect_performance_batch(params=self.theta, seed=self.seed+i*self.batch_size)
            self.performance_idx_theta[i, :] = base_perf_batch
            self.performance_idx[i] = np.mean(base_perf_batch)

            self.update_best_theta(current_perf=self.performance_idx[i], params=self.theta)
            self.update_best_rho(current_perf=self.performance_idx[i])

            estimated_gradient = self.estimate_fd_gradient(base_mean_perf=self.performance_idx[i], seed=self.seed+i*self.batch_size)
            self.fd_grad_history[i, :] = copy.deepcopy(estimated_gradient)

            self.update_rho(estimated_gradient=estimated_gradient)

            self.theta_history[self.time, :] = copy.deepcopy(self.theta)
            self.rho_history[self.time, :] = copy.deepcopy(self.theta)

            self.time += 1
            if self.verbose:
                print(f"[PGPE-FD] step: {self.time}")
                print(f"[PGPE-FD] performance: {self.performance_idx[i]}")
                print(f"[PGPE-FD] gradient norm: {np.linalg.norm(estimated_gradient)}")

            if self.time % self.checkpoint_freq == 0 and self.directory is not None:
                self.save_results()

            if not self.learn_std:
                self.fd_eps = np.clip(self.fd_eps - self.std_decay, self.std_min, np.inf)
                self.rho[RhoElem.STD, :] = copy.deepcopy(self.fd_eps)

        if self.save_det:
            self.sample_deterministic_curve()

    def update_best_rho(self, current_perf: float, *args, **kwargs) -> None:
        """Update the best rho-like configuration found so far."""
        if current_perf > self.best_performance_rho:
            self.best_rho = copy.deepcopy(self.rho)
            self.best_performance_rho = current_perf
            print("-" * 30)
            print(f"New best RHO: {self.best_rho}")
            print(f"New best PERFORMANCE: {self.best_performance_rho}")
            print("-" * 30)

            if self.directory is not None:
                if self.directory != "":
                    file_name = self.directory + "/best_rho"
                else:
                    file_name = "best_rho"
                np.save(file_name, self.best_rho)

    def update_best_theta(self, current_perf: float, params: np.array, *args, **kwargs) -> None:
        """Update the best policy parameter found so far."""
        if current_perf > self.best_performance_theta:
            self.best_theta = copy.deepcopy(np.array(params, dtype=np.float64))
            self.best_performance_theta = current_perf

            if self.directory is not None:
                if self.directory != "":
                    file_name = self.directory + "/best_theta"
                else:
                    file_name = "best_theta"
                np.save(file_name, self.best_theta)

    def sample_deterministic_curve(self):
        """Compute deterministic performance of parameter history."""
        for i in tqdm(range(self.ite)):
            self.policy.set_parameters(thetas=self.theta_history[i, :])
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=IdentityDataProcessor(),
                params=None,
                starting_state=None
            )
            delayed_functions = delayed(pg_sampling_worker)
            res = Parallel(n_jobs=self.n_jobs_param, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(self.batch_size)
            )

            ite_perf = np.zeros(self.batch_size, dtype=np.float64)
            for j in range(self.batch_size):
                ite_perf[j] = res[j][TrajectoryResults.PERF]

            self.deterministic_curve[i] = np.mean(ite_perf)

    def save_results(self) -> None:
        """Save training results."""
        results = {
            "performance": np.array(self.performance_idx, dtype=float).tolist(),
            "performance_thetas_per_rho": np.array(self.performance_idx_theta, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "best_rho": np.array(self.best_rho, dtype=float).tolist(),
            "thetas_history": np.array(self.theta_history, dtype=float).tolist(),
            "rho_history": np.array(self.rho_history, dtype=float).tolist(),
            "fd_eps": np.array(self.fd_eps, dtype=float).tolist(),
            "fd_grad_history": np.array(self.fd_grad_history, dtype=float).tolist(),
            "deterministic_res": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        name = self.directory + "/results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
