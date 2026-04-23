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

from policies.nn_policy import NeuralNetworkPolicy, to_torch
import torch


class PolicyGradient:
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
            fd_action_eps: float = 1e-3,
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

        # Sigma for action perturbation and theta perturbation in finite difference estimation
        self.fd_action_eps = fd_action_eps

    
    # ── Support methods ────────────────────────────────────────────────────────

    def _compute_tail_returns(self, rewards: np.ndarray) -> np.ndarray:
        tail = np.zeros(len(rewards), dtype=np.float64)
        running = 0
        for i in reversed(range(len(rewards))):
            running = rewards[i] + self.env.gamma * running
            tail[i] = running
        return tail

    def _sample_perturbations(self, horizon: int) -> np.ndarray:
        if self.perturbation_scope == "trajectory":
            eps = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(1, self.dim_action)
            ).astype(np.float64)
        else:
            eps = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(horizon, self.dim_action)
            ).astype(np.float64)

        norms = np.linalg.norm(eps, axis=1, keepdims=True)
        zero_norm_rows = norms.squeeze(-1) == 0
        while np.any(zero_norm_rows):
            eps[zero_norm_rows] = np.random.normal(
                loc=0.0,
                scale=1.0,
                size=(np.sum(zero_norm_rows), self.dim_action)
            ).astype(np.float64)
            norms = np.linalg.norm(eps, axis=1, keepdims=True)
            zero_norm_rows = norms.squeeze(-1) == 0

        eps = eps / norms
        if self.perturbation_scope == "trajectory":
            eps = np.repeat(eps, horizon, axis=0)

        return eps

    def _set_env_state(self, env_obj, state):
        if hasattr(env_obj, "set_state"):
            env_obj.set_state(copy.deepcopy(state))
        else:
            env_obj.state = copy.deepcopy(state)

    
    # ── Trajectory sampling methods ────────────────────────────────────────────────────────

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

    def _paired_rollout_forward(
            self,
            params: np.ndarray,
            eps_traj: np.ndarray,
            seed: int,
            starting_state: np.ndarray = None
    ) -> dict:
        env_nom = copy.deepcopy(self.env)
        env_pert = copy.deepcopy(self.env)
        pol_nom = copy.deepcopy(self.policy)
        pol_pert = copy.deepcopy(self.policy)
        dp_nom = copy.deepcopy(self.data_processor)
        dp_pert = copy.deepcopy(self.data_processor)

        env_nom.reset(seed=seed)
        env_pert.reset(seed=seed)
        if starting_state is not None:
            self._set_env_state(env_nom, starting_state)
            self._set_env_state(env_pert, starting_state)

        pol_nom.set_parameters(thetas=copy.deepcopy(params))
        pol_pert.set_parameters(thetas=copy.deepcopy(params))

        rewards_nom = []
        rewards_pert = []
        features_nom = []
        features_pert = []
        eps_used = []

        perf_nom = 0.0
        perf_pert = 0.0

        for t in range(self.env.horizon):
            state_nom = copy.deepcopy(env_nom.state)
            state_pert = copy.deepcopy(env_pert.state)

            # Can be removed
            feat_nom = np.array(dp_nom.transform(state=state_nom), dtype=np.float64)
            feat_pert = np.array(dp_pert.transform(state=state_pert), dtype=np.float64)

            # Sample action
            mu_nom = np.atleast_1d(np.array(pol_nom.draw_action(state=feat_nom), dtype=np.float64))
            mu_pert = np.atleast_1d(np.array(pol_pert.draw_action(state=feat_pert), dtype=np.float64))

            # Apply perturbation 
            eps_t = np.atleast_1d(eps_traj[t])
            action_nom = mu_nom
            action_pert = mu_pert + self.fd_action_eps * eps_t

            # Step envs
            _, rew_nom, done_nom, _ = env_nom.step(action=action_nom)
            _, rew_pert, done_pert, _ = env_pert.step(action=action_pert)
            
            # Update discounted performance
            perf_nom += (self.env.gamma ** t) * rew_nom
            perf_pert += (self.env.gamma ** t) * rew_pert

            rewards_nom.append(rew_nom)
            rewards_pert.append(rew_pert)

            features_nom.append(feat_nom)
            features_pert.append(feat_pert)
            eps_used.append(eps_t)

            if done_nom or done_pert:
                break

        return {
            "perf_nom": float(perf_nom),
            "perf_pert": float(perf_pert),
            "rewards_nom": np.array(rewards_nom, dtype=np.float64),
            "rewards_pert": np.array(rewards_pert, dtype=np.float64),
            "features_nom": np.array(features_nom, dtype=np.float64),
            "features_pert": np.array(features_pert, dtype=np.float64),
            "eps": np.array(eps_used, dtype=np.float64)
        }

    def _paired_rollout_central(
            self,
            params: np.ndarray,
            eps_traj: np.ndarray,
            seed: int,
            starting_state: np.ndarray = None
    ) -> dict:
        env_nom = copy.deepcopy(self.env)
        env_plus = copy.deepcopy(self.env)
        env_minus = copy.deepcopy(self.env)
        pol_nom = copy.deepcopy(self.policy)
        pol_plus = copy.deepcopy(self.policy)
        pol_minus = copy.deepcopy(self.policy)
        dp_nom = copy.deepcopy(self.data_processor)
        dp_plus = copy.deepcopy(self.data_processor)
        dp_minus = copy.deepcopy(self.data_processor)

        env_nom.reset(seed=seed)
        env_plus.reset(seed=seed)
        env_minus.reset(seed=seed)

        if starting_state is not None:
            self._set_env_state(env_nom, starting_state)
            self._set_env_state(env_plus, starting_state)
            self._set_env_state(env_minus, starting_state)

        pol_nom.set_parameters(thetas=copy.deepcopy(params))
        pol_plus.set_parameters(thetas=copy.deepcopy(params))
        pol_minus.set_parameters(thetas=copy.deepcopy(params))

        rewards_nom = []
        rewards_plus = []
        rewards_minus = []
        features_nom = []
        features_plus = []
        features_minus = []
        eps_used = []

        perf_nom = 0.0

        for t in range(self.env.horizon):
            state_nom = copy.deepcopy(env_nom.state)
            state_plus = copy.deepcopy(env_plus.state)
            state_minus = copy.deepcopy(env_minus.state)

            feat_nom = np.array(dp_nom.transform(state=state_nom), dtype=np.float64)
            feat_plus = np.array(dp_plus.transform(state=state_plus), dtype=np.float64)
            feat_minus = np.array(dp_minus.transform(state=state_minus), dtype=np.float64)

            mu_nom = np.atleast_1d(np.array(pol_nom.draw_action(state=feat_nom), dtype=np.float64))
            mu_plus = np.atleast_1d(np.array(pol_plus.draw_action(state=feat_plus), dtype=np.float64))
            mu_minus = np.atleast_1d(np.array(pol_minus.draw_action(state=feat_minus), dtype=np.float64))

            eps_t = np.atleast_1d(eps_traj[t])
            action_nom = mu_nom
            action_plus = mu_plus + self.fd_action_eps * eps_t
            action_minus = mu_minus - self.fd_action_eps * eps_t

            _, rew_nom, done_nom, _ = env_nom.step(action=action_nom)
            _, rew_plus, done_plus, _ = env_plus.step(action=action_plus)
            _, rew_minus, done_minus, _ = env_minus.step(action=action_minus)

            perf_nom += (self.env.gamma ** t) * rew_nom

            rewards_nom.append(rew_nom)
            rewards_plus.append(rew_plus)
            rewards_minus.append(rew_minus)
            features_nom.append(feat_nom)
            features_plus.append(feat_plus)
            features_minus.append(feat_minus)
            eps_used.append(eps_t)

            if done_nom or done_plus or done_minus:
                break

        return {
            "perf_nom": float(perf_nom),
            "rewards_nom": np.array(rewards_nom, dtype=np.float64),
            "rewards_plus": np.array(rewards_plus, dtype=np.float64),
            "rewards_minus": np.array(rewards_minus, dtype=np.float64),
            "features_nom": np.array(features_nom, dtype=np.float64),
            "features_plus": np.array(features_plus, dtype=np.float64),
            "features_minus": np.array(features_minus, dtype=np.float64),
            "eps": np.array(eps_used, dtype=np.float64)
        }

    def _rollout_deterministic_set(
            self,
            params: np.ndarray,
            seed: int,
            starting_state: np.ndarray = None
    ) -> dict:
        env_nom = copy.deepcopy(self.env)
        pol_nom = copy.deepcopy(self.policy)
        dp_nom = copy.deepcopy(self.data_processor)

        env_nom.reset(seed=seed)
        if starting_state is not None:
            self._set_env_state(env_nom, starting_state)
        pol_nom.set_parameters(thetas=copy.deepcopy(params))

        rewards_nom = []
        features_nom = []
        perf_nom = 0.0

        for t in range(self.env.horizon):
            state_nom = copy.deepcopy(env_nom.state)
            feat_nom = np.array(dp_nom.transform(state=state_nom), dtype=np.float64)
            mu_nom = np.atleast_1d(np.array(pol_nom.draw_action(state=feat_nom), dtype=np.float64))

            _, rew_nom, done_nom, _ = env_nom.step(action=mu_nom)

            perf_nom += (self.env.gamma ** t) * rew_nom
            rewards_nom.append(rew_nom)
            features_nom.append(feat_nom)

            if done_nom:
                break

        rewards_perturbed = []
        perf_perturbed = []

        for i in range(self.dim_action):
            env_pert = copy.deepcopy(self.env)
            pol_pert = copy.deepcopy(self.policy)
            dp_pert = copy.deepcopy(self.data_processor)

            env_pert.reset(seed=seed)
            if starting_state is not None:
                self._set_env_state(env_pert, starting_state)
            pol_pert.set_parameters(thetas=copy.deepcopy(params))

            eps_i = np.zeros(self.dim_action, dtype=np.float64)
            eps_i[i] = 1.0

            rewards_i = []
            perf_i = 0.0

            for t in range(self.env.horizon):
                state_pert = copy.deepcopy(env_pert.state)
                feat_pert = np.array(dp_pert.transform(state=state_pert), dtype=np.float64)
                mu_pert = np.atleast_1d(np.array(pol_pert.draw_action(state=feat_pert), dtype=np.float64))
                action_pert = mu_pert + self.fd_action_eps * eps_i

                _, rew_pert, done_pert, _ = env_pert.step(action=action_pert)

                perf_i += (self.env.gamma ** t) * rew_pert
                rewards_i.append(rew_pert)

                if done_pert:
                    break

            rewards_perturbed.append(np.array(rewards_i, dtype=np.float64))
            perf_perturbed.append(float(perf_i))

        return {
            "perf_nom": float(perf_nom),
            "perf_perturbed": np.array(perf_perturbed, dtype=np.float64),
            "rewards_nom": np.array(rewards_nom, dtype=np.float64),
            "rewards_perturbed": rewards_perturbed,
            "features_nom": np.array(features_nom, dtype=np.float64)
        }

    def _rollout_deterministic_set_central(
            self,
            params: np.ndarray,
            seed: int,
            starting_state: np.ndarray = None
    ) -> dict:
        env_nom = copy.deepcopy(self.env)
        pol_nom = copy.deepcopy(self.policy)
        dp_nom = copy.deepcopy(self.data_processor)

        env_nom.reset(seed=seed)
        if starting_state is not None:
            self._set_env_state(env_nom, starting_state)
        pol_nom.set_parameters(thetas=copy.deepcopy(params))

        rewards_nom = []
        features_nom = []
        perf_nom = 0.0

        for t in range(self.env.horizon):
            state_nom = copy.deepcopy(env_nom.state)
            feat_nom = np.array(dp_nom.transform(state=state_nom), dtype=np.float64)
            mu_nom = np.atleast_1d(np.array(pol_nom.draw_action(state=feat_nom), dtype=np.float64))

            _, rew_nom, done_nom, _ = env_nom.step(action=mu_nom)

            perf_nom += (self.env.gamma ** t) * rew_nom
            rewards_nom.append(rew_nom)
            features_nom.append(feat_nom)

            if done_nom:
                break

        rewards_plus = []
        rewards_minus = []

        for i in range(self.dim_action):
            env_plus = copy.deepcopy(self.env)
            env_minus = copy.deepcopy(self.env)
            pol_plus = copy.deepcopy(self.policy)
            pol_minus = copy.deepcopy(self.policy)
            dp_plus = copy.deepcopy(self.data_processor)
            dp_minus = copy.deepcopy(self.data_processor)

            env_plus.reset(seed=seed)
            env_minus.reset(seed=seed)
            if starting_state is not None:
                self._set_env_state(env_plus, starting_state)
                self._set_env_state(env_minus, starting_state)
            pol_plus.set_parameters(thetas=copy.deepcopy(params))
            pol_minus.set_parameters(thetas=copy.deepcopy(params))

            eps_i = np.zeros(self.dim_action, dtype=np.float64)
            eps_i[i] = 1.0

            rewards_i_plus = []
            rewards_i_minus = []

            for t in range(self.env.horizon):
                state_plus = copy.deepcopy(env_plus.state)
                state_minus = copy.deepcopy(env_minus.state)
                feat_plus = np.array(dp_plus.transform(state=state_plus), dtype=np.float64)
                feat_minus = np.array(dp_minus.transform(state=state_minus), dtype=np.float64)
                mu_plus = np.atleast_1d(np.array(pol_plus.draw_action(state=feat_plus), dtype=np.float64))
                mu_minus = np.atleast_1d(np.array(pol_minus.draw_action(state=feat_minus), dtype=np.float64))

                action_plus = mu_plus + self.fd_action_eps * eps_i
                action_minus = mu_minus - self.fd_action_eps * eps_i

                _, rew_plus, done_plus, _ = env_plus.step(action=action_plus)
                _, rew_minus, done_minus, _ = env_minus.step(action=action_minus)

                rewards_i_plus.append(rew_plus)
                rewards_i_minus.append(rew_minus)

                if done_plus or done_minus:
                    break

            rewards_plus.append(np.array(rewards_i_plus, dtype=np.float64))
            rewards_minus.append(np.array(rewards_i_minus, dtype=np.float64))

        return {
            "perf_nom": float(perf_nom),
            "rewards_nom": np.array(rewards_nom, dtype=np.float64),
            "rewards_plus": rewards_plus,
            "rewards_minus": rewards_minus,
            "features_nom": np.array(features_nom, dtype=np.float64)
        }

    # ── NN version ────────────────────────────────────────────────────────

    def _is_nn_policy(self) -> bool:
        """Check if the current policy is an MLP-based NeuralNetworkPolicy."""
        return isinstance(self.policy, NeuralNetworkPolicy)

    def _build_surrogate_loss(self, paired_trajectories: list) -> "torch.Tensor":
        """
        Construct the scalar surrogate loss L such that:

            dL/dθ  =  (1/B·δ) Σ_b Σ_t  γ^t · [ J(s_t')ᵀ εt · Gt'  -  J(s_t)ᵀ εt · Gt ]

        where J(s) = ∂μ_θ(s)/∂θ.  G_t and ε_t enter only as detached scalars/vectors.
        Supports forward-difference trajectories (keys: features_nom / features_pert).
        """
        mlp = self.policy.mlp
        mlp.eval()  # no dropout / batchnorm stochasticity during Jacobian pass

        loss = torch.tensor(0.0, dtype=torch.float64)

        for traj in paired_trajectories:
            feats_nom   = traj["features_nom"]    # list[np.ndarray]  length T
            feats_pert  = traj["features_pert"]
            rewards_nom = traj["rewards_nom"]
            rewards_pert = traj["rewards_pert"]
            eps_seq     = traj["eps"]             # (T, dim_action)

            horizon_t = min(len(rewards_nom), len(rewards_pert), len(eps_seq))
            if horizon_t == 0:
                continue

            tail_nom  = self._compute_tail_returns(rewards_nom[:horizon_t])   # Gt
            tail_pert = self._compute_tail_returns(rewards_pert[:horizon_t])  # Gt'

            for t in range(horizon_t):
                # ── detached constants ──────────────────────────────────────────
                eps_t   = to_torch(eps_seq[t]).detach()       # (dim_action,)
                G_t_nom  = float(tail_nom[t])
                G_t_pert = float(tail_pert[t])
                discount = self.env.gamma ** t

                # ── MLP forward (graph kept) ────────────────────────────────────
                s_nom  = to_torch(feats_nom[t])   # (dim_state,)
                s_pert = to_torch(feats_pert[t])

                mu_nom  = mlp(s_nom)   # (dim_action,)  — differentiable
                mu_pert = mlp(s_pert)  # (dim_action,)  — differentiable

                # εᵀ μ(s) is a scalar; G_t scales it — grad gives J(s)ᵀ ε · G_t
                loss = loss + discount * (
                    torch.dot(eps_t, mu_pert) * G_t_pert
                    - torch.dot(eps_t, mu_nom)  * G_t_nom
                )

        # normalise by batch size and δ
        loss = loss / (self.batch_size * self.fd_action_eps)
        return loss


    def _estimate_fd_gradient_nn(self, paired_trajectories: list) -> "np.ndarray":
        """
        Gradient estimator for NeuralNetworkPolicy via surrogate-loss autograd.
        Drop-in replacement for _estimate_fd_gradient.
        """
        mlp = self.policy.mlp
        mlp.zero_grad()

        loss = self._build_surrogate_loss(paired_trajectories)
        loss.backward()

        # Flatten all parameter gradients into a single numpy vector
        grad_vec = torch.nn.utils.parameters_to_vector(
            [p.grad if p.grad is not None else torch.zeros_like(p)
            for p in mlp.parameters()]
        )
        return grad_vec.detach().cpu().numpy()


    def _build_surrogate_loss_deterministic(self, trajectories: list) -> "torch.Tensor":
        """
        Surrogate loss for the coordinate-wise deterministic FD estimator:

            g = (1/B) Σ_b Σ_t  γ^t · J(s_t)ᵀ · v_t

        where  v_t[i] = (G_t^i - G_t) / δ  for each action dimension i,
        obtained via dL/dθ with  L = (1/B) Σ_b Σ_t  γ^t · dot(v_t, μ_θ(s_t)).

        Expects trajectories from _rollout_deterministic_set (forward-diff variant).
        """
        mlp = self.policy.mlp
        mlp.eval()

        loss = torch.tensor(0.0, dtype=torch.float64)

        for traj in trajectories:
            feats_nom        = traj["features_nom"]       # list[np.ndarray], length T
            rewards_nom      = traj["rewards_nom"]         # (T,)
            rewards_perturbed = traj["rewards_perturbed"]  # list[dim_action] of (T,)

            # Align horizons across all perturbed arms
            horizon_t = len(rewards_nom)
            for i in range(self.dim_action):
                horizon_t = min(horizon_t, len(rewards_perturbed[i]))
            if horizon_t == 0:
                continue

            tail_nom = self._compute_tail_returns(rewards_nom[:horizon_t])          # G_t
            tail_pert = [
                self._compute_tail_returns(rewards_perturbed[i][:horizon_t])        # G_t^i
                for i in range(self.dim_action)
            ]

            for t in range(horizon_t):
                # ── detached FD weight vector v_t ──────────────────────────────
                v_t = torch.tensor(
                    [(tail_pert[i][t] - tail_nom[t]) / self.fd_action_eps
                    for i in range(self.dim_action)],
                    dtype=torch.float64
                ).detach()                                   # (dim_action,) — constant

                # ── MLP forward (graph kept) ───────────────────────────────────
                s_t   = to_torch(feats_nom[t])               # (dim_state,)
                mu_t  = mlp(s_t).double()                           # (dim_action,) — differentiable

                loss = loss + (self.env.gamma ** t) * torch.dot(v_t, mu_t)

        loss = loss / self.batch_size
        return loss


    def _estimate_fd_gradient_deterministic_nn(self, trajectories: list) -> "np.ndarray":
        """
        Drop-in NN replacement for _estimate_fd_gradient_deterministic.
        """
        mlp = self.policy.mlp
        mlp.zero_grad()

        loss = self._build_surrogate_loss_deterministic(trajectories)
        loss.backward()

        grad_vec = torch.nn.utils.parameters_to_vector(
            [p.grad if p.grad is not None else torch.zeros_like(p)
            for p in mlp.parameters()]
        )
        return grad_vec.detach().cpu().numpy()

    
    # ── Linear version ────────────────────────────────────────────────────────

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

    def _estimate_fd_gradient(self, paired_trajectories: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for traj in paired_trajectories:
            # Extract data from trajectory
            rewards_nom = traj["rewards_nom"]
            rewards_pert = traj["rewards_pert"]
            feats_nom = traj["features_nom"]
            feats_pert = traj["features_pert"]
            eps_seq = traj["eps"]

            if len(rewards_nom) == 0 or len(rewards_pert) == 0:
                continue
            
            horizon_t = min(len(rewards_nom), len(rewards_pert), len(eps_seq))
            if horizon_t == 0:
                continue
            
            # Utile????
            tail_nom = self._compute_tail_returns(rewards_nom[:horizon_t])
            tail_pert = self._compute_tail_returns(rewards_pert[:horizon_t])

            for t in range(horizon_t):
                # Compute policy Jacobian at time t for nominal and perturbed trajectories
                jac_nom = self._policy_jacobian(features=feats_nom[t])
                jac_pert = self._policy_jacobian(features=feats_pert[t])

                # Retrive perturbation at time t
                eps_t = np.atleast_1d(eps_seq[t])

                # Compute FD gradient contribution at time t
                term_nom = (jac_nom.T @ eps_t) * tail_nom[t]
                term_pert = (jac_pert.T @ eps_t) * tail_pert[t]

                estimated_gradient += (self.env.gamma ** t) * (term_pert - term_nom) / self.fd_action_eps
        # Mean over batch
        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def _estimate_fd_gradient_central(self, paired_trajectories: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for traj in paired_trajectories:
            rewards_plus = traj["rewards_plus"]
            rewards_minus = traj["rewards_minus"]
            feats_plus = traj["features_plus"]
            feats_minus = traj["features_minus"]
            eps_seq = traj["eps"]

            if len(rewards_plus) == 0 or len(rewards_minus) == 0:
                continue

            horizon_t = min(len(rewards_plus), len(rewards_minus), len(eps_seq))
            if horizon_t == 0:
                continue

            tail_plus = self._compute_tail_returns(rewards_plus[:horizon_t])
            tail_minus = self._compute_tail_returns(rewards_minus[:horizon_t])

            for t in range(horizon_t):
                jac_plus = self._policy_jacobian(features=feats_plus[t])
                jac_minus = self._policy_jacobian(features=feats_minus[t])

                eps_t = np.atleast_1d(eps_seq[t])

                term_plus = (jac_plus.T @ eps_t) * tail_plus[t]
                term_minus = (jac_minus.T @ eps_t) * tail_minus[t]

                estimated_gradient += (self.env.gamma ** t) * (term_plus - term_minus) / (2.0 * self.fd_action_eps)

        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def _estimate_fd_gradient_deterministic(self, trajectories: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for traj in trajectories:
            rewards_nom = traj["rewards_nom"]
            feats_nom = traj["features_nom"]
            rewards_perturbed = traj["rewards_perturbed"]

            if len(rewards_nom) == 0:
                continue

            horizon_t = len(rewards_nom)
            for i in range(self.dim_action):
                horizon_t = min(horizon_t, len(rewards_perturbed[i]))

            if horizon_t == 0:
                continue

            tail_nom = self._compute_tail_returns(rewards_nom[:horizon_t])
            tail_perturbed = [
                self._compute_tail_returns(rewards_perturbed[i][:horizon_t])
                for i in range(self.dim_action)
            ]

            for t in range(horizon_t):
                jac_nom = self._policy_jacobian(features=feats_nom[t])

                delta_returns = np.zeros(self.dim_action, dtype=np.float64)
                for i in range(self.dim_action):
                    delta_returns[i] = (tail_perturbed[i][t] - tail_nom[t]) / self.fd_action_eps

                estimated_gradient += (self.env.gamma ** t) * (jac_nom.T @ delta_returns)

        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def _estimate_fd_gradient_deterministic_central(self, trajectories: list) -> np.ndarray:
        estimated_gradient = np.zeros(self.dim, dtype=np.float64)

        for traj in trajectories:
            rewards_nom = traj["rewards_nom"]
            feats_nom = traj["features_nom"]
            rewards_plus = traj["rewards_plus"]
            rewards_minus = traj["rewards_minus"]

            if len(rewards_nom) == 0:
                continue

            horizon_t = len(rewards_nom)
            for i in range(self.dim_action):
                horizon_t = min(horizon_t, len(rewards_plus[i]), len(rewards_minus[i]))

            if horizon_t == 0:
                continue

            tail_plus = [
                self._compute_tail_returns(rewards_plus[i][:horizon_t])
                for i in range(self.dim_action)
            ]
            tail_minus = [
                self._compute_tail_returns(rewards_minus[i][:horizon_t])
                for i in range(self.dim_action)
            ]

            for t in range(horizon_t):
                jac_nom = self._policy_jacobian(features=feats_nom[t])

                delta_returns = np.zeros(self.dim_action, dtype=np.float64)
                for i in range(self.dim_action):
                    delta_returns[i] = (tail_plus[i][t] - tail_minus[i][t]) / (2.0 * self.fd_action_eps)

                estimated_gradient += (self.env.gamma ** t) * (jac_nom.T @ delta_returns)

        estimated_gradient = estimated_gradient / self.batch_size
        return estimated_gradient

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            if self.fd_rollout_mode == "stochastic":
                # np.random.seed(self.seed + i)

                # Sample from the unit sphere for each time step and each trajectory in the batch
                eps_batch = [self._sample_perturbations(self.env.horizon) for _ in range(self.batch_size)]

                # TODO check for loops
                if self.fd_mode == "forward":
                    if self.parallel_computation:
                        trajectories = Parallel(n_jobs=self.n_jobs, backend="loky")(
                            delayed(self._paired_rollout_forward)(
                                params=self.thetas,
                                eps_traj=eps_batch[b],
                                seed=self.seed+i*self.batch_size,
                                starting_state=self.starting_state
                            ) for b in range(self.batch_size)
                        )
                    else:
                        trajectories = []
                        for b in range(self.batch_size):
                            traj = self._paired_rollout_forward(
                                params=self.thetas,
                                eps_traj=eps_batch[b],
                                seed=self.seed+i*self.batch_size,
                                starting_state=self.starting_state
                            )
                            trajectories.append(traj)

                    if not self._is_nn_policy():
                        estimated_gradient = self._estimate_fd_gradient(paired_trajectories=trajectories)
                    else:
                        estimated_gradient = self._estimate_fd_gradient_nn(paired_trajectories=trajectories)

                elif self.fd_mode == "central":
                    if self.parallel_computation:
                        trajectories = Parallel(n_jobs=self.n_jobs, backend="loky")(
                            delayed(self._paired_rollout_central)(
                                params=self.thetas,
                                eps_traj=eps_batch[b],
                                seed=self.seed+i*self.batch_size,
                                starting_state=self.starting_state
                            ) for b in range(self.batch_size)
                        )
                    else:
                        trajectories = []
                        for b in range(self.batch_size):
                            traj = self._paired_rollout_central(
                                params=self.thetas,
                                eps_traj=eps_batch[b],
                                seed=self.seed+i*self.batch_size,
                                starting_state=self.starting_state
                            )
                            trajectories.append(traj)

                    estimated_gradient = self._estimate_fd_gradient_central(paired_trajectories=trajectories)
                else:
                    raise NotImplementedError("[PG-FD] five_point mode not implemented for stochastic rollout.")

            # Deterministic case ---> TODO remove useless methods and add determinism as a parameter in the rollout methods
            else:
                if self.fd_mode == "forward":
                    if self.parallel_computation:
                        trajectories = Parallel(n_jobs=self.n_jobs, backend="loky")(
                            delayed(self._rollout_deterministic_set)(
                                params=self.thetas,
                                seed=self.seed,
                                starting_state=self.starting_state
                            ) for b in range(self.batch_size)
                        )
                    else:
                        trajectories = []
                        for b in range(self.batch_size):
                            traj = self._rollout_deterministic_set(
                                params=self.thetas,
                                seed=self.seed,
                                starting_state=self.starting_state
                            )
                            trajectories.append(traj)

                    if not self._is_nn_policy():
                        estimated_gradient = self._estimate_fd_gradient_deterministic(paired_trajectories=trajectories)
                    else:
                        estimated_gradient = self._estimate_fd_gradient_deterministic_nn(trajectories=trajectories)
                elif self.fd_mode == "central":
                    if self.parallel_computation:
                        trajectories = Parallel(n_jobs=self.n_jobs, backend="loky")(
                            delayed(self._rollout_deterministic_set_central)(
                                params=self.thetas,
                                seed=self.seed,
                                starting_state=self.starting_state
                            ) for b in range(self.batch_size)
                        )
                    else:
                        trajectories = []
                        for b in range(self.batch_size):
                            traj = self._rollout_deterministic_set_central(
                                params=self.thetas,
                                seed=self.seed,
                                starting_state=self.starting_state
                            )
                            trajectories.append(traj)

                    estimated_gradient = self._estimate_fd_gradient_deterministic_central(trajectories=trajectories)
                else:
                    raise NotImplementedError("[PG-FD] five_point mode not implemented for deterministic rollout.")

            # Extract nominal performance for each trajectory and compute the mean performance across the batch
            perf_vector = np.array([traj["perf_nom"] for traj in trajectories], dtype=np.float64)

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
            "deterministic_res": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        name = self.directory + "/results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
