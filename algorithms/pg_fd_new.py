"""Policy Gradient finite-difference implementation"""

import copy
import io
import json

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from envs.base_env import BaseEnv
from envs.swimmer import Swimmer
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import check_directory_and_create, LearnRates
from adam.adam import Adam

from policies.nn_policy import NeuralNetworkPolicy, to_torch
import torch


class PolicyGradientFD_2:
    """Policy Gradient with action-space finite differences."""
    def __init__(
            self, lr: np.array = None,
            lr_strategy: str = "constant",
            estimator_type: str = "FD",
            ite: int = 100,
            batch_size: int = 1,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "",
            verbose: bool = False,
            natural: bool = False,
            checkpoint_freq: int = 1,
            n_jobs: int = 1,
            save_det: int = 0,
            seed: int = 0,
            fd_rollout_mode: str = "stochastic",
            starting_state: np.ndarray = None,
            fd_mode: str = "forward",
            perturbation_scope: str = "step",
            fd_action_delta: float = 1e-3,
    ) -> None:
        err_msg = "[PG-FD] lr must be positive!"
        assert lr[LearnRates.PARAM] > 0, err_msg
        self.lr = lr[LearnRates.PARAM]

        err_msg = "[PG-FD] lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy

        self.estimator_type = estimator_type

        err_msg = "[PG-FD] policy is None."
        assert policy is not None, err_msg
        self.policy = policy
        self.thetas = self.policy.get_parameters()
        self.dim = len(self.thetas)

        err_msg = "[PG-FD] env is None."
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PG-FD] data processor is None."
        assert data_processor is not None, err_msg
        self.data_processor = data_processor

        self.directory = directory
        if self.directory is not None:
            check_directory_and_create(dir_name=directory)
        self.save_det = save_det

        self.ite = ite
        self.batch_size = batch_size
        self.verbose = verbose
        self.natural = natural
        self.checkpoint_freq = checkpoint_freq
        self.n_jobs = n_jobs
        self.parallel_computation = bool(self.n_jobs != 1)

        self.dim_action = self.env.action_dim
        self.dim_state = self.env.state_dim

        self.theta_history = np.zeros((self.ite, self.dim), dtype=np.float64)
        self.fd_grad_history = np.zeros((self.ite, self.dim), dtype=np.float64)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_performance_theta = -np.inf
        self.deterministic_curve = np.zeros(self.ite)

        if self.ite > 0:
            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)

        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam(self.lr, strategy="ascent")

        self.std_score = None
        self.seed = seed

        if starting_state is None:
            env_obj = copy.deepcopy(self.env)
            env_obj.reset(seed=self.seed)
            self.starting_state = copy.deepcopy(env_obj.state)
        else:
            self.starting_state = copy.deepcopy(starting_state)

        err_msg = "[PG-FD] fd_rollout_mode not valid!"
        assert fd_rollout_mode in ["stochastic", "deterministic"], err_msg
        self.fd_rollout_mode = fd_rollout_mode

        err_msg = "[PG-FD] fd_mode not valid!"
        assert fd_mode in ["forward", "central", "five_point"], err_msg
        self.fd_mode = fd_mode

        err_msg = "[PG-FD] perturbation_scope not valid!"
        assert perturbation_scope in ["step", "trajectory"], err_msg
        self.perturbation_scope = perturbation_scope

        # delta for action perturbation
        self.fd_action_delta = fd_action_delta

    def _set_env_state(self, env_obj, state):
        if hasattr(env_obj, "set_state"):
            env_obj.set_state(copy.deepcopy(state))
        else:
            env_obj.state = copy.deepcopy(state)

    def sample_delta(self, horizon: int) -> np.ndarray:
        if self.perturbation_scope == "trajectory":
            eps = np.random.uniform(
                low=-1.0,
                high=1.0,
                size=(1, self.dim_action)
            ).astype(np.float64)
        else:
            eps = np.random.uniform(
                low=-1.0,
                high=1.0,
                size=(horizon, self.dim_action)
            ).astype(np.float64)

        norms = np.sum(eps ** 2, axis=1, keepdims=True) ** 0.5
        # norms = np.linalg.norm(eps, axis=1, keepdims=True)

        eps = eps / norms
        if self.perturbation_scope == "trajectory":
            eps = np.repeat(eps, horizon, axis=0)

        return eps

    def rollout(
            self,
            params: np.ndarray,
            eps_traj: np.ndarray,
            seed: int,
            starting_state: np.ndarray = None
    ) -> dict:

        env = copy.deepcopy(self.env)
        pol = copy.deepcopy(self.policy)

        env.reset(seed=seed)
        if starting_state is not None:
            self._set_env_state(env, starting_state)

        pol.set_parameters(thetas=copy.deepcopy(params))

        rewards = []
        features = []
        eps_used = []

        perf = 0.0

        for t in range(self.env.horizon):
            state_nom = copy.deepcopy(env.state)

            feat = np.array(state_nom, dtype=np.float64)

            # Sample action
            mu = np.atleast_1d(np.array(pol.draw_action(state=feat), dtype=np.float64))

            # Apply perturbation 
            eps_t = np.atleast_1d(eps_traj[t])
            action = mu + self.fd_action_delta * eps_t

            # Step envs
            _, rew, done, _ = env.step(action=action)
            
            # Update discounted performance
            perf += (self.env.gamma ** t) * rew

            rewards.append(rew)
            features.append(feat)
            eps_used.append(eps_t)

            if done:
                break   
        
        perf_to_go = np.zeros(len(rewards), dtype=np.float64)
        running = 0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.env.gamma * running
            perf_to_go[i] = running

        return {
            "perf": float(perf),
            "rewards": np.array(rewards, dtype=np.float64),
            "features": np.array(features, dtype=np.float64),
            "eps": np.array(eps_used, dtype=np.float64),
            "perf_to_go": perf_to_go
        }

    def _policy_jacobian(self, features: np.ndarray) -> np.ndarray:
        """
        Exact Jacobian computation for a linear policy mu(s) = Theta * s.
        Assumes self.thetas is a flattened array of shape (dim_action * dim_state).
        """
        dim_state = len(features)
        
        # Verify that the total parameter dimension matches the linear assumption
        err_msg = f"Parameter dimension mismatch. Expected {self.dim_action * dim_state}, got {self.dim}."
        assert self.dim == self.dim_action * dim_state, err_msg

        jac = np.zeros((self.dim_action, self.dim), dtype=np.float64)
        
        # Construct the block-diagonal structure of the Jacobian
        for j in range(self.dim_action):
            start_idx = j * dim_state
            end_idx = start_idx + dim_state
            jac[j, start_idx:end_idx] = features
            
        return jac  

    def _estimate_fd_gradient(self, nom_traj, pert_traj: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for traj_nom, traj_pert in zip(nom_traj, pert_traj):
            # Extract data from trajectory
            perf_to_go_nom = traj_nom["perf_to_go"]
            feats_nom = traj_nom["features"]
            
            perf_to_go_pert = traj_pert["perf_to_go"]
            feats_pert = traj_pert["features"]
            eps_seq_pert = traj_pert["eps"]

            horizon_t = min(len(perf_to_go_nom), len(perf_to_go_pert))
            if horizon_t == 0:
                continue
        
            for t in range(horizon_t):
                # Compute policy Jacobian at time t for nominal and perturbed trajectories
                jac_nom = self._policy_jacobian(features=feats_nom[t])
                jac_pert = self._policy_jacobian(features=feats_pert[t])

                # Retrive perturbation at time t
                eps_t = np.atleast_1d(eps_seq_pert[t])

                # Compute FD gradient contribution at time t
                term_nom = jac_nom * perf_to_go_nom[t]
                term_pert = jac_pert * perf_to_go_pert[t]

                estimated_gradient += (self.env.gamma ** t) * ((term_pert - term_nom).T @ eps_t) / self.fd_action_delta

        # Mean over batch
        estimated_gradient = estimated_gradient / self.batch_size

        return estimated_gradient
    
    def _estimate_fd_gradient_central(self, pert_traj_plus: list, pert_traj_minus: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for traj_plus, traj_minus in zip(pert_traj_plus, pert_traj_minus):
            perf_to_go_plus = traj_plus["perf_to_go"]
            feats_plus = traj_plus["features"]
            
            perf_to_go_minus = traj_minus["perf_to_go"]
            feats_minus = traj_minus["features"]
            eps_seq = traj_plus["eps"]

            horizon_t = min(len(perf_to_go_plus), len(perf_to_go_minus))
            if horizon_t == 0:
                continue
        
            for t in range(horizon_t):
                jac_plus = self._policy_jacobian(features=feats_plus[t])
                jac_minus = self._policy_jacobian(features=feats_minus[t])

                eps_t = np.atleast_1d(eps_seq[t])

                term_plus = jac_plus * perf_to_go_plus[t]
                term_minus = jac_minus * perf_to_go_minus[t]

                estimated_gradient += (self.env.gamma ** t) * ((term_plus - term_minus).T @ eps_t) / (2.0 * self.fd_action_delta)

        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def _estimate_fd_gradient_deterministic(self, nom_traj: list, pert_traj_list: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for b, traj_nom in enumerate(nom_traj):
            perf_to_go_nom = traj_nom["perf_to_go"]
            feats_nom = traj_nom["features"]
            pert_trajs = pert_traj_list[b]
            
            horizon_t = len(perf_to_go_nom)
            for i in range(self.dim_action):
                horizon_t = min(horizon_t, len(pert_trajs[i]["perf_to_go"]))

            if horizon_t == 0:
                continue
                
            for t in range(horizon_t):
                jac_nom = self._policy_jacobian(features=feats_nom[t])

                delta_returns = np.zeros(self.dim_action, dtype=np.float64)
                for i in range(self.dim_action):
                    delta_returns[i] = (pert_trajs[i]["perf_to_go"][t] - perf_to_go_nom[t]) / self.fd_action_delta
                
                estimated_gradient += (self.env.gamma ** t) * (jac_nom.T @ delta_returns)

        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def _estimate_fd_gradient_deterministic_central(self, nom_traj: list, pert_traj_plus_list: list, pert_traj_minus_list: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for b, traj_nom in enumerate(nom_traj):
            perf_to_go_nom = traj_nom["perf_to_go"]
            feats_nom = traj_nom["features"]
            trajs_plus = pert_traj_plus_list[b]
            trajs_minus = pert_traj_minus_list[b]
            
            horizon_t = len(perf_to_go_nom)
            for i in range(self.dim_action):
                horizon_t = min(horizon_t, len(trajs_plus[i]["perf_to_go"]), len(trajs_minus[i]["perf_to_go"]))

            if horizon_t == 0:
                continue
                
            for t in range(horizon_t):
                jac_nom = self._policy_jacobian(features=feats_nom[t])

                delta_returns = np.zeros(self.dim_action, dtype=np.float64)
                for i in range(self.dim_action):
                    delta_returns[i] = (trajs_plus[i]["perf_to_go"][t] - trajs_minus[i]["perf_to_go"][t]) / (2.0 * self.fd_action_delta)
                
                estimated_gradient += (self.env.gamma ** t) * (jac_nom.T @ delta_returns)

        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def learn(self):
        for i in tqdm(range(self.ite)):
            # Generate nominal trajectories
            nom_traj = Parallel(n_jobs=self.n_jobs)(
                delayed(self.rollout)(
                    params=copy.deepcopy(self.thetas),
                    eps_traj=np.zeros((self.env.horizon, self.dim_action), dtype=np.float64),
                    seed=self.seed+i*self.batch_size+b,
                    starting_state=self.starting_state
                ) for b in range(self.batch_size)
            )

            if self.fd_rollout_mode == "stochastic":
                eps_batch = [self.sample_delta(horizon=self.env.horizon) for _ in range(self.batch_size)]
                
                if self.fd_mode == "forward":
                    pert_traj = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.rollout)(
                            params=copy.deepcopy(self.thetas),
                            eps_traj=copy.deepcopy(eps_batch[b]),
                            seed=self.seed+i*self.batch_size+b,
                            starting_state=self.starting_state
                        ) for b in range(self.batch_size)
                    )
                    estimated_gradient = self._estimate_fd_gradient(nom_traj=nom_traj, pert_traj=pert_traj)
                elif self.fd_mode == "central":
                    pert_traj_plus = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.rollout)(
                            params=copy.deepcopy(self.thetas),
                            eps_traj=copy.deepcopy(eps_batch[b]),
                            seed=self.seed+i*self.batch_size+b,
                            starting_state=self.starting_state
                        ) for b in range(self.batch_size)
                    )
                    pert_traj_minus = Parallel(n_jobs=self.n_jobs)(
                        delayed(self.rollout)(
                            params=copy.deepcopy(self.thetas),
                            eps_traj=-copy.deepcopy(eps_batch[b]),
                            seed=self.seed+i*self.batch_size+b,
                            starting_state=self.starting_state
                        ) for b in range(self.batch_size)
                    )
                    estimated_gradient = self._estimate_fd_gradient_central(pert_traj_plus, pert_traj_minus)
                else:
                    raise NotImplementedError("Only forward and central are implemented for stochastic FD.")

            elif self.fd_rollout_mode == "deterministic":
                if self.fd_mode == "forward":
                    pert_traj_list = []
                    for b in range(self.batch_size):
                        pert_trajs_i = []
                        for act_i in range(self.dim_action):
                            eps_i = np.zeros(self.dim_action, dtype=np.float64)
                            eps_i[act_i] = 1.0
                            eps_traj = np.ones((self.env.horizon, 1)) * eps_i
                            
                            traj = self.rollout(
                                params=copy.deepcopy(self.thetas),
                                eps_traj=eps_traj,
                                seed=self.seed+i*self.batch_size+b, # Maintain seed per batch element
                                starting_state=self.starting_state
                            )
                            pert_trajs_i.append(traj)
                        pert_traj_list.append(pert_trajs_i)
                    estimated_gradient = self._estimate_fd_gradient_deterministic(nom_traj, pert_traj_list)
                elif self.fd_mode == "central":
                    pert_traj_plus_list = []
                    pert_traj_minus_list = []
                    for b in range(self.batch_size):
                        trajs_plus_i = []
                        trajs_minus_i = []
                        for act_i in range(self.dim_action):
                            eps_i = np.zeros(self.dim_action, dtype=np.float64)
                            eps_i[act_i] = 1.0
                            eps_traj = np.ones((self.env.horizon, 1)) * eps_i
                            
                            traj_plus = self.rollout(
                                params=copy.deepcopy(self.thetas),
                                eps_traj=eps_traj,
                                seed=self.seed+i*self.batch_size+b,
                                starting_state=self.starting_state
                            )
                            traj_minus = self.rollout(
                                params=copy.deepcopy(self.thetas),
                                eps_traj=-eps_traj,
                                seed=self.seed+i*self.batch_size+b,
                                starting_state=self.starting_state
                            )
                            trajs_plus_i.append(traj_plus)
                            trajs_minus_i.append(traj_minus)
                        pert_traj_plus_list.append(trajs_plus_i)
                        pert_traj_minus_list.append(trajs_minus_i)
                    estimated_gradient = self._estimate_fd_gradient_deterministic_central(nom_traj, pert_traj_plus_list, pert_traj_minus_list)
                else:
                    raise NotImplementedError("Only forward and central are implemented for deterministic FD.")
            
            # Extract nominal performance for each trajectory and compute the mean performance across the batch

            perf_vector = np.array([traj["perf"] for traj in nom_traj], dtype=np.float64)

            self.performance_idx[i] = np.mean(perf_vector)
            self.update_best_theta(current_perf=self.performance_idx[i])

            self.fd_grad_history[i, :] = copy.deepcopy(estimated_gradient)

            if self.lr_strategy == "constant":
                self.thetas = self.thetas + self.lr * estimated_gradient
            elif self.lr_strategy == "adam":
                adaptive_lr = self.adam_optimizer.compute_gradient(estimated_gradient)
                self.thetas = self.thetas + adaptive_lr
            else:
                err_msg = f"[PG-FD] {self.lr_strategy} not implemented yet!"
                raise NotImplementedError(err_msg)

            if self.verbose:
                print("*" * 30)
                print(f"Step: {self.time}")
                print(f"Mean Performance: {self.performance_idx[i]}")
                print(f"Estimated gradient: {estimated_gradient}")
                print(f"Parameter (new) values: {self.thetas}")
                print(f"Best performance so far: {self.best_performance_theta}")
                print(f"Best configuration so far: {self.best_theta}")
                print("*" * 30)

            if self.time % self.checkpoint_freq == 0 and self.directory is not None:
                self.save_results()

            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)
            self.time += 1

        if self.save_det:
            self.sample_deterministic_curve()



    def update_best_theta(self, current_perf: np.float64, *args, **kwargs) -> None:
        """Updates the best theta configuration."""
        if self.best_theta is None or self.best_performance_theta <= current_perf:
            self.best_performance_theta = current_perf
            self.best_theta = copy.deepcopy(self.thetas)

            print("#" * 30)
            print("New best parameter configuration found")
            print(f"Performance: {self.best_performance_theta}")
            print(f"Parameter configuration: {self.best_theta}")
            print("#" * 30)

    def sample_deterministic_curve(self):
        """Collect deterministic performance over theta history."""
        saved_std_dev = None
        saved_sigma_noise = None
        if hasattr(self.policy, "std_dev"):
            saved_std_dev = copy.deepcopy(self.policy.std_dev)
            self.policy.std_dev = 0
        if hasattr(self.policy, "sigma_noise"):
            saved_sigma_noise = copy.deepcopy(self.policy.sigma_noise)
            try:
                self.policy.sigma_noise = 0
            except Exception:
                self.policy.sigma_noise = np.array(0)

        for i in tqdm(range(self.ite)):
            perf_batch = np.zeros(self.batch_size, dtype=np.float64)
            for j in range(self.batch_size):
                perf_batch[j] = self._rollout(
                    params=self.theta_history[i, :],
                    return_trace=False,
                    seed=self.seed + j + i * self.batch_size
                )
            self.deterministic_curve[i] = np.mean(perf_batch)

        if saved_std_dev is not None:
            self.policy.std_dev = saved_std_dev
        if saved_sigma_noise is not None:
            self.policy.sigma_noise = saved_sigma_noise

    def save_results(self) -> None:
        """Save the results."""
        results = {
            "performance": np.array(self.performance_idx, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "thetas_history": np.array(self.theta_history, dtype=float).tolist(),
            "last_theta": np.array(self.thetas, dtype=float).tolist(),
            "best_perf": float(self.best_performance_theta),
            "fd_gradient_history": np.array(self.fd_grad_history, dtype=float).tolist(),
            "deterministic_res": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        name = self.directory + "/results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()


# if __name__ == "__main__":
#     pg_fd = PolicyGradientFD_2(
#         lr = np.array([0.01], dtype=np.float64),
#         lr_strategy = "adam",
#         estimator_type = "FD",
#         ite = 500,
#         batch_size = 10,
#         env = Swimmer(horizon=200, gamma=1, render=False, clip=bool(0)),
#         policy 


#     lr: np.array = None,
#             lr_strategy: str = "constant",
#             estimator_type: str = "FD",
#             ite: int = 100,
#             batch_size: int = 1,
#             env: BaseEnv = None,
#             policy: BasePolicy = None,
#             data_processor: BaseProcessor = IdentityDataProcessor(),
#             directory: str = "",
#             verbose: bool = False,
#             natural: bool = False,
#             checkpoint_freq: int = 1,
#             n_jobs: int = 1,
#             save_det: int = 0,
#             seed: int = 0,
#             fd_rollout_mode: str = "stochastic",
#             starting_state: np.ndarray = None,
#             fd_mode: str = "forward",
#             perturbation_scope: str = "step",
#             fd_action_delta: float = 1e-3,