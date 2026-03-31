"""Policy Gradient finite-difference implementation"""

import copy
import io
import json

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import check_directory_and_create, LearnRates
from adam.adam import Adam


class PolicyGradient:
    """Policy Gradient with action-space finite differences."""
    def __init__(
            self, lr: np.array = None,
            lr_strategy: str = "constant",
            estimator_type: str = "FD",
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
            n_jobs: int = 1,
            save_det: int = 0,
                learn_std: int = 0,
                seed: int = 0
    ) -> None:
        err_msg = "[PG-FD] lr must be positive!"
        assert lr[LearnRates.PARAM] > 0, err_msg
        self.lr = lr[LearnRates.PARAM]

        err_msg = "[PG-FD] lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy

        self.estimator_type = estimator_type

        err_msg = "[PG-FD] initial_theta has not been specified!"
        assert initial_theta is not None, err_msg
        self.thetas = np.array(initial_theta, dtype=np.float64)
        self.dim = len(self.thetas)

        err_msg = "[PG-FD] env is None."
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PG-FD] policy is None."
        assert policy is not None, err_msg
        self.policy = policy

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

        self.learn_std = learn_std
        self.std_score = None
        self.seed = seed

        self.fd_action_eps = 1e-3
        self.fd_theta_eps = 1e-4

    def _set_env_state(self, env_obj, state):
        if hasattr(env_obj, "set_state"):
            env_obj.set_state(copy.deepcopy(state))
        else:
            env_obj.state = copy.deepcopy(state)

    def _rollout(
            self,
            params: np.ndarray,
            start_state=None,
            start_action=None,
            max_horizon: int = None,
            return_trace: bool = False,
            seed: int = 0
    ):
        env_obj = copy.deepcopy(self.env)
        pol_obj = copy.deepcopy(self.policy)
        dp_obj = copy.deepcopy(self.data_processor)

        env_obj.reset(seed=seed)
        np.random.seed(seed)
        if start_state is not None:
            self._set_env_state(env_obj, start_state)

        pol_obj.set_parameters(thetas=copy.deepcopy(params))

        if max_horizon is None:
            horizon = env_obj.horizon
        else:
            horizon = int(max_horizon)

        perf = 0
        states = []
        features = []
        actions = []
        rewards = []

        for t in range(horizon):
            state = copy.deepcopy(env_obj.state)
            feat = dp_obj.transform(state=state)

            if t == 0 and start_action is not None:
                action = np.array(start_action, dtype=np.float64)
            else:
                action = np.array(pol_obj.draw_action(state=feat), dtype=np.float64)
            action = np.atleast_1d(action)

            _, rew, done, _ = env_obj.step(action=action)
            perf += (env_obj.gamma ** t) * rew

            if return_trace:
                states.append(state)
                features.append(np.array(feat, dtype=np.float64))
                actions.append(copy.deepcopy(action))
                rewards.append(rew)

            if done:
                break

        if return_trace:
            return {
                "perf": perf,
                "states": states,
                "features": features,
                "actions": actions,
                "rewards": np.array(rewards, dtype=np.float64)
            }
        return perf

    def _compute_tail_returns(self, rewards: np.ndarray) -> np.ndarray:
        tail = np.zeros(len(rewards), dtype=np.float64)
        running = 0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.env.gamma * running
            tail[i] = running
        return tail

    def _policy_jacobian(self, features: np.ndarray) -> np.ndarray:
        saved_theta = copy.deepcopy(self.thetas)

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

        try:
            jac = np.zeros((self.dim_action, self.dim), dtype=np.float64)
            for p_idx in range(self.dim):
                theta_plus = copy.deepcopy(saved_theta)
                theta_minus = copy.deepcopy(saved_theta)
                theta_plus[p_idx] += self.fd_theta_eps
                theta_minus[p_idx] -= self.fd_theta_eps

                self.policy.set_parameters(thetas=theta_plus)
                action_plus = np.atleast_1d(np.array(self.policy.draw_action(state=features), dtype=np.float64))

                self.policy.set_parameters(thetas=theta_minus)
                action_minus = np.atleast_1d(np.array(self.policy.draw_action(state=features), dtype=np.float64))

                jac[:, p_idx] = (action_plus - action_minus) / (2 * self.fd_theta_eps)
        finally:
            self.policy.set_parameters(thetas=saved_theta)
            if saved_std_dev is not None:
                self.policy.std_dev = saved_std_dev
            if saved_sigma_noise is not None:
                self.policy.sigma_noise = saved_sigma_noise

        return jac

    def _collect_perturbed_q_values(self, trajectories: list, iteration_idx: int) -> list:
        perturbed_q_values = []

        for traj in trajectories:
            rewards = traj["rewards"]
            perturbed_q_values.append(np.zeros((len(rewards), self.dim_action), dtype=np.float64))

        rollout_inputs = []
        rollout_meta = []

        for traj_idx, traj in enumerate(trajectories):
            states = traj["states"]
            actions = traj["actions"]
            rewards = traj["rewards"]

            for t in range(len(rewards)):
                state_t = states[t]
                action_t = np.atleast_1d(actions[t])

                remaining_horizon = self.env.horizon - t
                if remaining_horizon <= 0:
                    continue

                for a_idx in range(self.dim_action):
                    pert_action = copy.deepcopy(action_t)
                    pert_action[a_idx] += self.fd_action_eps

                    local_seed = (
                        self.seed
                        + iteration_idx * self.batch_size * self.env.horizon * self.dim_action
                        + traj_idx * self.env.horizon * self.dim_action
                        + t * self.dim_action
                        + a_idx
                    )
                    rollout_inputs.append(dict(
                        params=self.thetas,
                        start_state=state_t,
                        start_action=pert_action,
                        max_horizon=remaining_horizon,
                        return_trace=False,
                        seed=local_seed
                    ))
                    rollout_meta.append((traj_idx, t, a_idx))

        if len(rollout_inputs) == 0:
            return perturbed_q_values

        if self.parallel_computation:
            rollout_results = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(self._rollout)(**kwargs) for kwargs in rollout_inputs
            )
        else:
            rollout_results = [self._rollout(**kwargs) for kwargs in rollout_inputs]

        for idx, perf in enumerate(rollout_results):
            traj_idx, t, a_idx = rollout_meta[idx]
            perturbed_q_values[traj_idx][t, a_idx] = perf

        return perturbed_q_values

    def _estimate_fd_gradient(self, trajectories: list, perturbed_q_values: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for traj_idx, traj in enumerate(trajectories):
            states = traj["states"]
            feats = traj["features"]
            rewards = traj["rewards"]

            if len(rewards) == 0:
                continue

            tail_returns = self._compute_tail_returns(rewards)
            traj_q = perturbed_q_values[traj_idx]

            for t in range(len(rewards)):
                feat_t = feats[t]

                grad_q_action = np.zeros(self.dim_action, dtype=np.float64)
                for a_idx in range(self.dim_action):
                    pert_q = traj_q[t, a_idx]
                    grad_q_action[a_idx] = (pert_q - tail_returns[t]) / self.fd_action_eps

                jac_t = self._policy_jacobian(features=feat_t)
                estimated_gradient += (self.env.gamma ** t) * (jac_t.T @ grad_q_action)

        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            if self.parallel_computation:
                trajectories = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed(self._rollout)(
                        params=self.thetas,
                        return_trace=True,
                        seed=self.seed + b + i * self.batch_size
                    ) for b in range(self.batch_size)
                )
            else:
                trajectories = []
                for b in range(self.batch_size):
                    traj = self._rollout(
                        params=self.thetas,
                        return_trace=True,
                        seed=self.seed + b + i * self.batch_size
                    )
                    trajectories.append(traj)

            perf_vector = np.array([traj["perf"] for traj in trajectories], dtype=np.float64)

            self.performance_idx[i] = np.mean(perf_vector)
            self.update_best_theta(current_perf=self.performance_idx[i])

            perturbed_q_values = self._collect_perturbed_q_values(
                trajectories=trajectories,
                iteration_idx=i
            )
            estimated_gradient = self._estimate_fd_gradient(
                trajectories=trajectories,
                perturbed_q_values=perturbed_q_values
            )
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

            # self.policy.reduce_exploration()

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
            "performance_det": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        name = self.directory + "/pg_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
