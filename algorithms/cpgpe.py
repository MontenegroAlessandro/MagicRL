"""
Implementation of C-PGPE.
"""
# Libraries
import numpy as np
from copy import deepcopy
from algorithms.pgpe import PGPE
import io
import json
from tqdm import tqdm
from adam.adam import Adam
from algorithms.utils import LearnRates, check_directory_and_create, ParamSamplerResults
from data_processors import IdentityDataProcessor
from algorithms.samplers import *


class CPGPE(PGPE):
    """CPGPE implementation"""
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
            deterministic: bool = False
    ):
        """
        Summary:
            Initialization function.

        Args:
            cost_type (str): type of cost to be selected among the macro "cost_types";
            cost_param (ndarray): the parameters for the cost measures (when necessary);
            omega (float): the ridge-regularization term for the Lagrangian, defaults to 0;
            thresholds (ndarray): the thresholds for the constraints;
            lambda_init (ndarray): the initial values for the lagrangian multipliers;
            eta_init (ndarray): the initial values for the additional learning variable;
            alternate (bool): flag telling if alternate optimization or not.

            all the other parameters have the same meaning as shown in PGPE.
        """
        # Super-class initialization
        super(CPGPE, self).__init__(
            lr=lr, initial_rho=initial_rho, ite=ite, batch_size=batch_size,
            episodes_per_theta=episodes_per_theta, env=env, policy=policy,
            data_processor=data_processor, directory=directory, verbose=verbose, natural=natural,
            checkpoint_freq=checkpoint_freq, lr_strategy=lr_strategy, learn_std=learn_std,
            std_decay=std_decay, std_min=std_min, n_jobs_param=n_jobs_param, n_jobs_traj=n_jobs_traj
        )
        # New parameters
        self.n_constraints = self.env.how_many_costs
        self.alternate = alternate

        # cost type
        err_msg = f"[CPGPE] Cost type {cost_type} not in {CPGPE.cost_types}."
        assert cost_type in CPGPE.cost_types, err_msg
        self.cost_type = cost_type

        # the additional cost parameter (not always needed)
        err_msg = f"[CPGPE] cost_param must be >= 0 ({cost_param} provided)."
        assert cost_param >= 0 and len(cost_param) == self.n_constraints, err_msg
        self.cost_param = cost_param

        # regularization term
        err_msg = f"[CPGPE] omega must be >= 0 ({omega} provided)."
        assert omega >= 0, err_msg
        self.omega = omega

        # thresholds for constraints
        err_msg = f"[CPGPE] wrong number of thresholds ({len(thresholds)} provided)."
        assert len(thresholds) == self.n_constraints, err_msg
        self.thresholds = np.array(thresholds, dtype=np.float64)

        # lambda and eta
        if lambda_init is not None:
            err_msg = f"[CPGPE] lambda_init has an incorrect length ({len(lambda_init)} provided)."
            assert len(lambda_init) == self.n_constraints, err_msg
            self.lambdas = np.array(lambda_init, dtype=np.float64)
        else:
            self.lambdas = np.zeros(self.n_constraints, dtype=np.float64)
        if eta_init is not None:
            err_msg = f"[CPGPE] eta_init has an incorrect length ({len(eta_init)} provided)."
            assert len(eta_init) == self.n_constraints, err_msg
            self.etas = np.array(eta_init, dtype=np.float64)
        else:
            self.etas = np.zeros(self.n_constraints, dtype=np.float64)

        # Modify already set fields
        # Remark: adam is computed always as ascent, then we use it with + or -.
        err_msg = f"[CPGPE] 3 step sizes needed ({len(lr)} provided)."
        assert len(lr) == 3, err_msg
        self.lr_rho = lr[LearnRates.PARAM]
        self.lr_lambda = lr[LearnRates.LAMBDA]
        self.lr_eta = lr[LearnRates.ETA]
        if self.lr_strategy == "adam":
            self.lambda_adam = Adam(step_size=self.lr_lambda, strategy="ascent")
            self.eta_adam = Adam(step_size=self.lr_eta, strategy="descent")

        # Utility fields
        self.use_eta = bool(self.cost_type not in ["tc", "chance"])
        self.cost_idx = np.zeros(shape=(self.ite, self.n_constraints), dtype=np.float64)
        self.cost_idx_theta = np.zeros(
            shape=(self.ite, self.batch_size, self.n_constraints),
            dtype=np.float64
        )
        self.risk_idx = np.zeros(shape=(self.ite, self.n_constraints), dtype=np.float64)
        self.risk_idx_theta = np.zeros(
            shape=(self.ite, self.batch_size, self.n_constraints),
            dtype=np.float64
        )
        self.deterministic_perf_curve = np.zeros(self.ite, dtype=np.float64)
        self.deterministic_cost_curve = np.zeros(
            shape=(self.ite, self.n_constraints),
            dtype=np.float64
        )
        self.lambda_history = np.zeros(shape=(self.ite, self.n_constraints), dtype=np.float64)
        self.lambda_history[0, :] = deepcopy(self.lambdas)
        self.eta_history = np.zeros((self.ite, self.n_constraints), dtype=np.float64)
        self.eta_history[0, :] = deepcopy(self.etas)

        self.deterministic = deterministic

        # Env check
        err_msg = f"[CPGPE] the provided env has not costs!"
        assert self.env.with_costs, err_msg

    def learn(self) -> None:
        """Learning Function"""
        for i in tqdm(range(self.ite)):
            # Collect trajectories
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

            # Post-process data
            performance_res = np.zeros(self.batch_size, dtype=np.float64)
            cost_res = np.zeros(shape=(self.batch_size, self.n_constraints), dtype=np.float64)
            for z in range(self.batch_size):
                self.thetas[z, :] = res[z][ParamSamplerResults.THETA]
                performance_res[z] = np.mean(res[z][ParamSamplerResults.PERF])
                cost_res[z, :] = np.mean(res[z][ParamSamplerResults.COST], axis=0)
            self.performance_idx_theta[i, :] = performance_res
            self.cost_idx_theta[i, :, :] = cost_res

            # Update performance
            self.performance_idx[i] = np.mean(self.performance_idx_theta[i, :])
            self.cost_idx[i, :] = np.mean(self.cost_idx_theta[i, :, :], axis=0)

            # compute risk measures
            self.compute_risks()

            # update best rho
            self.update_best_rho(current_perf=self.performance_idx[i], risks=self.risk_idx[i, :])

            # update best theta
            max_batch_perf = np.max(performance_res)
            best_theta_batch_index = np.where(performance_res == max_batch_perf)[0]
            self.update_best_theta(current_perf=self.performance_idx[i], params=self.thetas[best_theta_batch_index, :], risks=self.risk_idx[i, :])

            # Perform Alternate Ascent Descent Algorithm
            if self.alternate:
                if not (i % 2):
                    # update the rho vector
                    self.update_rho()

                    # (if needed) update the eta parameter
                    if self.use_eta:
                        self.update_eta()
                else:
                    # update lambda
                    self.update_lambda()
            # Perform Ascent Descent Algorithm
            else:
                self.update_rho()
                self.update_lambda()
                if self.use_eta:
                    self.update_eta()

            # Save the history of the parameters
            self.rho_history[self.time, :] = deepcopy(self.rho[RhoElem.MEAN])
            self.lambda_history[self.time, :] = deepcopy(self.lambdas)
            if self.use_eta:
                self.eta_history[self.time] = deepcopy(self.etas)

            # Update time counter
            self.time += 1

            # (when needed) print and save the info
            if self.verbose:
                print(f"rho perf: {self.performance_idx}")
                print(f"cost perf: {self.cost_idx}")
                print(f"theta perf: {self.performance_idx_theta}")
            if not (self.time % self.checkpoint_freq):
                self.save_results()

            # std_decay
            if not self.learn_std:
                std = np.float64(np.exp(self.rho[RhoElem.STD]))
                std = np.clip(std - self.std_decay, self.std_min, np.inf)
                self.rho[RhoElem.STD, :] = np.log(std)

        # Sample the deterministic curve
        if self.deterministic:
            self.sample_deterministic_curve()

        return

    def update_rho(self) -> None:
        # Compute the gradient
        m_grad_hat = None
        s_grad_hat = None

        # Take the performance of the whole batch: R(tau_i)
        batch_perf = self.performance_idx_theta[self.time, :]

        if self.cost_type in ["tc", "chance"]:
            batch_cum_cost = self.risk_idx_theta[self.time, :, :]
        elif self.cost_type == "cvar":
            batch_cum_cost = self.risk_idx_theta[self.time, :, :] - self.etas
        elif self.cost_type == "mv":
            batch_cum_cost = self.risk_idx_theta[self.time, :, :] - self.cost_param * np.power(self.etas, 2)
        else:
            raise NotImplementedError

        # Compute the batch cost
        batch_cost = np.sum(self.lambdas * batch_cum_cost, axis=1)

        # Combine costs and performances
        batch_mixed_index = - batch_perf + batch_cost

        # take the means and the sigmas
        means = self.rho[RhoElem.MEAN, :]
        stds = np.float64(np.exp(self.rho[RhoElem.STD, :]))

        # compute the scores
        if not self.natural:
            log_nu_means = (self.thetas - means) / (stds ** 2)
            log_nu_stds = (((self.thetas - means) ** 2) - (stds ** 2)) / (stds ** 2)
        else:
            log_nu_means = self.thetas - means
            log_nu_stds = (((self.thetas - means) ** 2) - (stds ** 2)) / (2 * (stds ** 2))

        # compute the gradients
        m_grad_hat = np.mean(batch_mixed_index[:, np.newaxis] * log_nu_means, axis=0)
        s_grad_hat = np.mean(batch_mixed_index[:, np.newaxis] * log_nu_stds, axis=0)

        # update the variable
        if self.lr_strategy == "constant":
            self.rho[RhoElem.MEAN, :] = self.rho[RhoElem.MEAN, :] - self.lr_rho * m_grad_hat
            if self.learn_std:
                self.rho[RhoElem.STD, :] = self.rho[RhoElem.STD, :] - self.lr * s_grad_hat
        elif self.lr_strategy == "adam":
            self.rho[RhoElem.MEAN, :] = self.rho[RhoElem.MEAN, :] - self.rho_adam[RhoElem.MEAN].compute_gradient(m_grad_hat)
            if self.learn_std:
                self.rho[RhoElem.STD, :] = self.rho[RhoElem.STD, :] - self.rho_adam[RhoElem.STD].compute_gradient(s_grad_hat)
        else:
            raise NotImplementedError

        return

    def update_lambda(self) -> None:
        # Compute the gradient
        grad_hat = self.risk_idx[self.time, :] - self.thresholds - self.omega * self.lambdas

        # update the variable
        if self.lr_strategy == "constant":
            self.lambdas = np.clip(self.lambdas + self.lr_lambda * grad_hat, 0, np.inf)
        elif self.lr_strategy == "adam":
            self.lambdas = np.clip(self.lambdas + self.lambda_adam.compute_gradient(grad_hat), 0, np.inf)
        else:
            raise NotImplementedError
        return

    def update_eta(self) -> None:
        # Compute the gradient
        if self.cost_type in ["tc", "chance"]:
            return
        elif self.cost_type == "cvar":
            grad_hat = self.lambdas - self.lambdas * np.power(1 - self.cost_param, -1) * np.mean(
                np.array(self.cost_idx_theta[self.time, :, :] >= self.etas, dtype=np.float64),
                axis=0
            )
        elif self.cost_type == "mv":
            grad_hat = np.mean(
                2 * self.cost_param * self.lambdas * (self.etas - self.cost_idx_theta[self.time, :, :]),
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

    def compute_risks(self) -> None:
        if self.cost_type == "tc":
            tmp_risk = deepcopy(self.cost_idx_theta[self.time, :, :])
        elif self.cost_type == "cvar":
            tmp_risk = np.clip(
                a=deepcopy(self.cost_idx_theta[self.time, :, :]) - self.etas,
                a_min=0,
                a_max=np.inf
            )
            tmp_risk = tmp_risk * np.power(1 - self.cost_param, -1) + self.etas
        elif self.cost_type == "mv":
            tmp_risk = self.cost_param * np.power(self.etas, 2)
            tmp_risk = tmp_risk + (1 - 2 * self.cost_param * self.etas) * self.cost_idx_theta[self.time, :, :]
            tmp_risk = tmp_risk + self.cost_param * np.power(self.cost_idx_theta[self.time, :, :], 2)
        elif self.cost_type == "chance":
            tmp_risk = np.array(
                self.cost_idx_theta[self.time, :, :] >= self.cost_param,
                dtype=np.float64
            )
        else:
            raise NotImplementedError(f"[CPGPE] {self.cost_type} not expected.")
        # save
        self.risk_idx_theta[self.time, :, :] = deepcopy(tmp_risk)
        self.risk_idx[self.time, :] = np.mean(self.risk_idx_theta[self.time, :, :], axis=0)

    def update_best_rho(
            self, current_perf: float, risks: np.array = None, *args, **kwargs
    ) -> None:
        """
        Save the best value of theta, that is the one in which all the constraints are respected
        """
        violation = risks - self.thresholds
        query = np.where(violation > 0)[0]
        if (current_perf > self.best_performance_rho) and (len(query) == 0):
            self.best_rho = deepcopy(self.rho)
            self.best_performance_rho = current_perf

            msg_1 = f"[CPGPE] New best RHO: {self.best_rho[RhoElem.MEAN]}"
            msg_2 = f"[CPGPE] New best PERFORMANCE: {self.best_performance_rho}"
            msg_3 = f"[CPGPE] CONSTRAINT VIOLATION: {violation}"
            max_len = max([len(msg_1), len(msg_2), len(msg_3)])

            print("#" * (max_len + 2))
            print("* " + msg_1)
            print("* " + msg_2)
            print("* " + msg_3)
            print("#" * (max_len + 2))

            # Save the best rho configuration
            if self.directory != "":
                file_name = self.directory + "/best_rho"
            else:
                file_name = "best_rho"
            np.save(file_name, self.best_rho)

    def update_best_theta(
            self, current_perf: float, params: np.ndarray, risks: np.ndarray = None,
            *args, **kwargs
    ) -> None:
        """
        Save the best value of theta, when the current performance is higher than the best performance.
        """
        violation = risks - self.thresholds
        if current_perf > self.best_performance_theta:
            self.best_theta = params
            self.best_performance_theta = current_perf

            msg_1 = f"[CPGPE] New best THETA: {self.best_theta}"
            msg_2 = f"[CPGPE] New best PERFORMANCE: {self.best_performance_theta}"
            msg_3 = f"[CPGPE] CONSTRAINT VIOLATION: {violation}"
            max_len = max([len(msg_1), len(msg_2), len(msg_3)])

            print("#" * (max_len + 2))
            print("* " + msg_1)
            print("* " + msg_2)
            print("* " + msg_3)
            print("#" * (max_len + 2))

            # Save the best theta configuration
            if self.directory != "":
                file_name = self.directory + "/best_theta"
            else:
                file_name = "best_theta"
            np.save(file_name, self.best_theta)

    def save_results(self) -> None:
        """Function saving the results of the training procedure"""
        # Create the dictionary with the useful info
        results = {
            "performance_rho": np.array(self.performance_idx, dtype=float).tolist(),
            "costs_rho": np.array(self.cost_idx, dtype=float).tolist(),
            "risks_rho": np.array(self.risk_idx, dtype=float).tolist(),
            #"performance_thetas_per_rho": np.array(self.performance_idx_theta, dtype=float).tolist(),
            # "costs_thetas_per_rho": np.array(self.cost_idx_theta, dtype=float).tolist(),
            # "risks_thetas_per_rho": np.array(self.risk_idx_theta, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "best_rho": np.array(self.best_rho, dtype=float).tolist(),
            #"thetas_history": np.array(self.thetas, dtype=float).tolist(),
            #"rho_history": np.array(self.rho_history, dtype=float).tolist(),
            "lambda_history": np.array(self.lambda_history, dtype=float).tolist(),
            #"eta_history": np.array(self.eta_history, dtype=float).tolist(),
            "deterministic_perf_res": np.array(self.deterministic_perf_curve, dtype=float).tolist(),
            "deterministic_cost_res": np.array(self.deterministic_cost_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/cpgpe_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return

    def sample_deterministic_curve(self):
        """
        Summary:
            Switch-off the noise and collect the deterministic performance
            associated to the sequence of parameter configurations seen during
            the learning.
        """
        # make the policy deterministic
        self.policy.std_dev = 0
        self.policy.sigma_noise = 0

        for i in tqdm(range(self.ite)):
            self.policy.set_parameters(thetas=copy.deepcopy(self.rho_history[i, :]))
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=self.data_processor,
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
            ite_cost = np.zeros((self.batch_size, self.n_constraints), dtype=np.float64)

            for j in range(self.batch_size):
                ite_perf[j] = res[j][TrajectoryResults.PERF]
                ite_cost[j] = res[j][TrajectoryResults.CostInfo]["cost_perf"]

            # compute mean
            self.deterministic_perf_curve[i] = np.mean(ite_perf)
            self.deterministic_cost_curve[i, :] = np.mean(ite_cost)
