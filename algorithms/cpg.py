"""
Implementation of C-PG (with GPOMDP estimator).
"""
import copy

# Libraries
import numpy as np
from copy import deepcopy
from algorithms.policy_gradient import PolicyGradient
import io
import json
from tqdm import tqdm
from adam.adam import Adam
from algorithms.utils import LearnRates, check_directory_and_create, ParamSamplerResults
from data_processors import IdentityDataProcessor
from algorithms.samplers import *


class CPolicyGradient(PolicyGradient):
    """CPG implementation"""
    # Macro for cost types
    cost_types = ["tc", "cvar", "mv", "chance"]
    # tc -> expected cost on the trajectories
    # cvar -> conditioned value at risk (requires the parameter)
    # mv -> mean variance (requires the parameter)
    # chance -> probability over the cost of trajectories (requires the parameter)

    def __init__(
            self,
            cost_type: str = "tc",
            cost_param: np.ndarray = None,
            omega: float = 0,
            thresholds: np.ndarray = None,
            lambda_init: np.ndarray = None,
            eta_init: np.ndarray = None,
            alternate: bool = True,
            lr: np.ndarray = None,
            lr_strategy: str = "constant",
            estimator_type: str = "REINFORCE",
            initial_theta: np.ndarray = None,
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
            deterministic: bool = False
    ):
        """
        Summary:
            Class initialization.

        Args:
            cost_type (str): type of cost to be selected among the macro "cost_types";
            cost_param (ndarray): the parameters for the cost measures (when necessary);
            omega (float): the ridge-regularization term for the Lagrangian, defaults to 0;
            thresholds (ndarray): the thresholds for the constraints;
            lambda_init (ndarray): the initial values for the lagrangian multipliers;
            eta_init (ndarray): the initial values for the additional learning variable;
            alternate (bool): flag telling if alternate optimization or not.

            all the other parameters have the same meaning as shown in PolicyGradient.
        """
        super().__init__(
            lr=lr,
            lr_strategy=lr_strategy,
            estimator_type=estimator_type,
            initial_theta=initial_theta,
            ite=ite,
            batch_size=batch_size,
            env=env,
            policy=policy,
            data_processor=data_processor,
            directory=directory,
            verbose=verbose,
            natural=natural,
            checkpoint_freq=checkpoint_freq,
            n_jobs=n_jobs
        )
        # New parameters
        self.alternate = alternate
        self.n_constraints = self.env.how_many_costs

        # cost type
        err_msg = f"[CPG] Cost type {cost_type} not in {CPolicyGradient.cost_types}."
        assert cost_type in CPolicyGradient.cost_types, err_msg
        self.cost_type = cost_type

        # the additional cost parameter (not always needed)
        err_msg = f"[CPG] cost_param must be >= 0 ({cost_param} provided)."
        assert cost_param >= 0 and len(cost_param) == self.n_constraints, err_msg
        self.cost_param = cost_param

        # regularization term
        err_msg = f"[CPG] omega must be >= 0 ({omega} provided)."
        assert omega >= 0, err_msg
        self.omega = omega

        # thresholds for constraints
        err_msg = f"[CPG] wrong number of thresholds ({len(thresholds)} provided)."
        assert len(thresholds) == self.n_constraints, err_msg
        self.thresholds = np.array(thresholds, dtype=np.float64)

        # lambda and eta
        if lambda_init is not None:
            err_msg = f"[CPG] lambda_init has an incorrect length ({len(lambda_init)} provided)."
            assert len(lambda_init) == self.n_constraints, err_msg
            self.lambdas = np.array(lambda_init, dtype=np.float64)
        else:
            self.lambdas = np.zeros(self.n_constraints, dtype=np.float64)
        if eta_init is not None:
            err_msg = f"[CPG] eta_init has an incorrect length ({len(eta_init)} provided)."
            assert len(eta_init) == self.n_constraints, err_msg
            self.etas = np.array(eta_init, dtype=np.float64)
        else:
            self.etas = np.zeros(self.n_constraints, dtype=np.float64)

        # Modify already set fields
        # Remark: adam is computed always as ascent, then we use it with + or -.
        err_msg = f"[CPG] 3 step sizes needed ({len(lr)} provided)."
        assert len(lr) == 3, err_msg
        self.lr_theta = lr[LearnRates.PARAM]
        self.lr_lambda = lr[LearnRates.LAMBDA]
        self.lr_eta = lr[LearnRates.ETA]
        if self.lr_strategy == "adam":
            self.theta_adam = Adam(step_size=self.lr_theta, strategy="descent")
            self.lambda_adam = Adam(step_size=self.lr_lambda, strategy="ascent")
            self.eta_adam = Adam(step_size=self.lr_eta, strategy="descent")

        # Utility fields
        self.use_eta = bool(self.cost_type not in ["tc", "chance"])
        self.cost_idx = np.zeros(shape=(self.ite, self.n_constraints), dtype=np.float64)
        """self.cost_idx_theta = np.zeros(
            shape=(self.ite, self.batch_size, self.n_constraints),
            dtype=np.float64
        )"""
        self.risk_idx = np.zeros(shape=(self.ite, self.n_constraints), dtype=np.float64)
        """self.risk_idx_theta = np.zeros(
            shape=(self.ite, self.batch_size, self.n_constraints),
            dtype=np.float64
        )"""
        self.deterministic_perf_curve = np.zeros(self.ite, dtype=np.float64)
        self.deterministic_cost_curve = np.zeros(
            shape=(self.ite, self.n_constraints),
            dtype=np.float64
        )
        self.lambda_history = np.zeros(shape=(self.ite, self.n_constraints), dtype=np.float64)
        self.lambda_history[0, :] = deepcopy(self.lambdas)
        self.eta_history = np.zeros((self.ite, self.n_constraints), dtype=np.float64)
        self.eta_history[0, :] = deepcopy(self.etas)

        self.determimnistic = deterministic

        # Env check
        err_msg = f"[CPGPE] the provided env has not costs!"
        assert self.env.with_costs, err_msg

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            if self.parallel_computation:
                # prepare the parameters
                self.policy.set_parameters(copy.deepcopy(self.thetas))
                worker_dict = dict(
                    env=copy.deepcopy(self.env),
                    pol=copy.deepcopy(self.policy),
                    dp=copy.deepcopy(self.data_processor),
                    # params=copy.deepcopy(self.thetas),
                    params=None,
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
            perf_vector = np.zeros(self.batch_size, dtype=np.float64)
            cost_perf_vector = np.zeros(
                shape=(self.batch_size, self.n_constraints),
                dtype=np.float64
            )
            score_vector = np.zeros(
                shape=(self.batch_size, self.env.horizon, self.dim),
                dtype=np.float64
            )
            reward_vector = np.zeros(shape=(self.batch_size, self.env.horizon), dtype=np.float64)
            cost_vector = np.zeros(
                shape=(self.batch_size, self.env.horizon, self.n_constraints),
                dtype=np.float64
            )
            for j in range(self.batch_size):
                perf_vector[j] = res[j][TrajectoryResults.PERF]
                cost_perf_vector[j] = np.array(
                    res[j][TrajectoryResults.CostInfo]["cost_perf"],
                    dtype=np.float64
                )
                reward_vector[j, :] = res[j][TrajectoryResults.RewList]
                score_vector[j, :, :] = res[j][TrajectoryResults.ScoreList]
                cost_vector[j, :, :] = np.array(
                    res[j][TrajectoryResults.CostInfo]["costs"],
                    dtype=np.float64
                )
            self.performance_idx[i] = np.mean(perf_vector)
            self.cost_idx[i, :] = np.mean(cost_perf_vector, axis=0)

            # compute risks
            risk_vector = self.compute_risk(cost_batch=copy.deepcopy(cost_perf_vector))
            self.risk_idx[i, :] = np.mean(risk_vector, axis=0)

            # Update best theta
            self.update_best_theta(
                current_perf=self.performance_idx[i],
                current_risk=self.risk_idx[i, :]
            )

            # update the parameters
            if self.alternate:
                if not (i % 2):
                    self.update_theta(
                        reward_trajectory=reward_vector,
                        score_trajectory=score_vector,
                        costs_trajectory=cost_vector
                    )
                    if self.use_eta:
                        self.update_eta(cost_perf=cost_perf_vector)
                else:
                    self.update_lambda()
            else:
                self.update_theta(
                    reward_trajectory=reward_vector,
                    score_trajectory=score_vector,
                    costs_trajectory=cost_vector
                )
                self.update_lambda()
                if self.use_eta:
                    self.update_eta(cost_perf=cost_perf_vector)

            # save learning targets
            self.theta_history[i, :] = copy.deepcopy(self.thetas)
            self.lambda_history[i, :] = copy.deepcopy(self.lambdas)
            self.eta_history[i, :] = copy.deepcopy(self.etas)

            # checkpoint
            if self.time % self.checkpoint_freq == 0:
                self.save_results()

            # save theta history
            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)

            # time update
            self.time += 1

            # reduce the exploration factor of the policy
            self.policy.reduce_exploration()

        if self.determimnistic:
            self.sample_deterministic_curve()
        return

    def update_theta(
            self,
            reward_trajectory,
            score_trajectory,
            costs_trajectory
    ) -> None:
        """Theta is computed with GPOMDP estimator when possible, elsewhere REINFORCE is used."""
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(score_trajectory, axis=1)

        reward_trajectory = reward_trajectory[:, :, np.newaxis] * rolling_scores
        rew_gpomdp = np.mean(
            np.sum(gamma_seq[:, np.newaxis] * reward_trajectory, axis=1),
            axis=0
        )

        if self.cost_type == "tc":
            costs_trajectory = np.sum(self.lambdas * costs_trajectory, axis=2)[:, :, np.newaxis] * rolling_scores
            estimated_gradient = np.mean(
                np.sum(gamma_seq[:, np.newaxis] * (-reward_trajectory + costs_trajectory), axis=1),
                axis=0
            )
        elif self.cost_type == "cvar":
            costs_trajectory = np.mean(costs_trajectory, axis=1)
            costs_trajectory = np.clip(a=costs_trajectory - self.etas, a_min=0, a_max=np.inf)
            costs_trajectory = costs_trajectory * self.lambdas * np.power(1 - self.cost_param, -1)
            costs_trajectory = np.sum(costs_trajectory, axis=1)
            costs_trajectory = costs_trajectory[:, np.newaxis] * np.sum(score_trajectory, axis=1)
            estimated_gradient = - rew_gpomdp + np.mean(costs_trajectory, axis=0)
        elif self.cost_type == "mv":
            """costs_trajectory = np.mean(costs_trajectory, axis=1)
            costs_trajectory = (1 - 2 * self.cost_param * self.etas) * costs_trajectory + self.cost_param * np.power(costs_trajectory, 2)
            costs_trajectory = np.sum(self.lambdas * costs_trajectory, axis=1)
            costs_trajectory = costs_trajectory[:, np.newaxis] * np.sum(score_trajectory, axis=1)
            estimated_gradient = - rew_gpomdp + np.mean(costs_trajectory, axis=0)"""
            sum_costs = np.sum(self.lambdas * costs_trajectory, axis=2)[:, :, np.newaxis] * rolling_scores
            # half_hat = np.mean(
            #     np.sum(gamma_seq[:, np.newaxis] * (-reward_trajectory + sum_costs), axis=1),
            #     axis=0
            # )
            half_hat = np.mean(
                np.sum(gamma_seq[:, np.newaxis] * (-reward_trajectory + (1 - 2 * self.cost_param * self.etas) * sum_costs), axis=1),
                axis=0
            )
            sq_costs_trajectory = self.cost_param * np.power(np.mean(costs_trajectory, axis=1), 2)
            sq_costs_trajectory = np.sum(self.lambdas * sq_costs_trajectory, axis=1)[:, np.newaxis] * np.sum(score_trajectory, axis=1)
            estimated_gradient = half_hat + np.mean(sq_costs_trajectory, axis=0)
        elif self.cost_type == "chance":
            # take the means over the horizon
            costs_trajectory = np.mean(costs_trajectory, axis=1)
            # indicator function
            costs_trajectory = np.array(costs_trajectory >= self.cost_param, dtype=np.float64)
            # reinforce
            costs_trajectory =  np.sum(self.lambdas * costs_trajectory, axis=1)[:, np.newaxis] * np.sum(score_trajectory, axis=1)
            # compute gradient
            estimated_gradient = - rew_gpomdp + np.mean(costs_trajectory, axis=0)
        else:
            raise NotImplementedError(f"[CPG] {self.cost_type} risk measure not implemented.")

        # update the variable
        if self.lr_strategy == "constant":
            self.thetas = self.thetas - self.lr_theta * estimated_gradient
        elif self.lr_strategy == "adam":
            self.thetas = self.thetas - self.theta_adam.compute_gradient(estimated_gradient)
        else:
            raise NotImplementedError
        return

    def update_lambda(self) -> None:
        # Compute the gradient
        grad_hat = self.risk_idx[self.time, :] - self.thresholds - self.omega * self.lambdas

        # update the variable
        if self.lr_strategy == "constant":
            self.lambdas = np.clip(a=self.lambdas + self.lr_lambda * grad_hat, a_min=0, a_max=np.inf)
        elif self.lr_strategy == "adam":
            self.lambdas = np.clip(a=self.lambdas + self.lambda_adam.compute_gradient(grad_hat), a_min=0, a_max=np.inf)
        else:
            raise NotImplementedError
        return

    def update_eta(self, cost_perf: np.ndarray) -> None:
        # Compute the gradient
        if self.cost_type in ["tc", "chance"]:
            return
        elif self.cost_type == "cvar":
            grad_hat = self.lambdas - self.lambdas * np.power(1 - self.cost_param, -1) * np.mean(
                np.array(cost_perf >= self.etas, dtype=np.float64),
                axis=0
            )
        elif self.cost_type == "mv":
            grad_hat = np.mean(
                2 * self.cost_param * self.lambdas * (self.etas - cost_perf),
                axis=0
            )
        else:
            raise NotImplementedError

        # update the variable
        if self.lr_strategy == "constant":
            self.etas = self.etas - self.lr_eta * grad_hat
        elif self.lr_strategy == "adam":
            self.etas = self.etas - self.eta_adam.compute_gradient(grad_hat)
        else:
            raise NotImplementedError
        return

    def compute_risk(self, cost_batch: np.ndarray = None) -> np.ndarray:
        if self.cost_type == "tc":
            risk = deepcopy(cost_batch)
        elif self.cost_type == "cvar":
            risk = np.clip(
                a=deepcopy(cost_batch) - self.etas,
                a_min=0,
                a_max=np.inf
            )
            risk = risk * np.power(1 - self.cost_param, -1) + self.etas
        elif self.cost_type == "mv":
            risk = self.cost_param * np.power(self.etas, 2)
            risk = risk + (1 - 2 * self.cost_param * self.etas) * cost_batch
            risk = risk + self.cost_param * np.power(cost_batch, 2)
        elif self.cost_type == "chance":
            risk = np.array(
                cost_batch >= self.cost_param,
                dtype=np.float64
            )
        else:
            raise NotImplementedError(f"[CPGPE] {self.cost_type} not expected.")
        return risk

    def sample_deterministic_curve(self) -> None:
        """
                Summary:
                    Switch-off the noise and collect the deterministic performance
                    associated to the sequence of parameter configurations seen during
                    the learning.
                """
        # make the policy deterministic
        self.policy.std_dev = 0
        self.policy.sigma_noise = 0

        # sample
        for i in tqdm(range(self.ite)):
            self.policy.set_parameters(thetas=self.theta_history[i, :])
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=self.data_processor,
                # params=copy.deepcopy(self.theta_history[i, :]),
                params=None,
                starting_state=None
            )
            # build the parallel functions
            delayed_functions = delayed(pg_sampling_worker)

            # parallel computation
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(self.batch_size)
            )
            # extract data
            ite_perf = np.zeros(self.batch_size, dtype=np.float64)
            ite_cost = np.zeros((self.batch_size, self.n_constraints), dtype=np.float64)

            for j in range(self.batch_size):
                ite_perf[j] = res[j][TrajectoryResults.PERF]
                ite_cost[j] = res[j][TrajectoryResults.CostInfo]["cost_perf"]

            # compute mean
            self.deterministic_perf_curve[i] = np.mean(ite_perf)
            self.deterministic_cost_curve[i, :] = np.mean(ite_cost)


    def update_best_theta(
            self,
            current_perf: np.float64,
            current_risk: np.ndarray = None,
            *args,
            **kwargs
    ) -> None:
        violation = current_risk - self.thresholds
        query = np.where(violation > 0)[0]
        if (self.best_theta is None or self.best_performance_theta <= current_perf) and (len(query) == 0):
            self.best_performance_theta = current_perf
            self.best_theta = copy.deepcopy(self.thetas)

            msg_1 = f"[CPG] New best THETA: {self.best_theta}"
            msg_2 = f"[CPG] New best PERFORMANCE: {self.best_performance_theta}"
            msg_3 = f"[CPG] CONSTRAINT VIOLATION: {violation}"
            max_len = max([len(msg_1), len(msg_2), len(msg_3)])

            print("#" * (max_len + 2))
            print("* " + msg_1)
            print("* " + msg_2)
            print("* " + msg_3)
            print("#" * (max_len + 2))
        return

    def save_results(self) -> None:
        """Save the results."""
        results = {
            "performance": np.array(self.performance_idx, dtype=float).tolist(),
            "costs": np.array(self.cost_idx, dtype=float).tolist(),
            "risks": np.array(self.risk_idx, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            #"thetas_history": np.array(self.theta_history, dtype=float).tolist(),
            "lambda_history": np.array(self.lambda_history, dtype=float).tolist(),
            #"eta_history": np.array(self.eta_history, dtype=float).tolist(),
            #"last_theta": np.array(self.thetas, dtype=float).tolist(),
            #"best_perf": float(self.best_performance_theta),
            "deterministic_perf_res": np.array(self.deterministic_perf_curve, dtype=float).tolist(),
            "deterministic_cost_res": np.array(self.deterministic_cost_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/cpg_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return