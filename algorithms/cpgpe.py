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
            cost_param: float = 0,
            omega: float = 0,
            thresholds: np.ndarray = None,
            lambda_init: np.ndarray = None,
            eta_init: float = 0,
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
            n_jobs_traj: int = 1
    ):
        """
        Summary:
            Initialization function.

        Args:
            cost_type (str): type of cost to be selected among the macro "cost_types";
            cost_param (float): the parameter for the cost measure (when necessary);
            omega (float): the ridge-regularization term for the Lagrangian, defaults to 0;
            thresholds (ndarray): the thresholds for the constraints;
            lambda_init (ndarray): the initial values for the lagrangian multipliers;
            eta_init (float): the initial value for the additional learning variable.

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
        # cost type
        err_msg = f"[CPGPE] Cost type {cost_type} not in {CPGPE.cost_types}."
        assert cost_type in CPGPE.cost_types, err_msg
        self.cost_type = cost_type

        # the additional cost parameter (not always needed)
        err_msg = f"[CPGPE] cost_param must be >= 0 ({cost_param} provided)."
        assert cost_param >= 0, err_msg
        self.cost_param = cost_param

        # regularization term
        err_msg = f"[CPGPE] omega must be >= 0 ({omega} provided)."
        assert omega >= 0, err_msg
        self.omega = omega

        # thresholds for constraints
        self.thresholds = np.array(thresholds, dtype=np.float64)
        self.n_constraints = len(self.thresholds)

        # lambda and eta
        if lambda_init is not None:
            err_msg = f"[CPGPE] lambda_init has an incorrect length ({len(lambda_init)} provided)."
            assert len(lambda_init) == self.n_constraints, err_msg
            self.lambdas = np.array(lambda_init, dtype=np.float64)
        else:
            self.lambdas = np.zeros(self.n_constraints, dtype=np.float64)
        self.eta = eta_init

        # Modify already set fields
        # fixme: fix the rho lr adam to descent
        err_msg = f"[CPGPE] 3 step sizes needed ({len(lr)} provided)."
        assert len(lr) == 3, err_msg
        self.lr_rho = lr[LearnRates.RHO]
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
        # todo: we may need the ones for the risks or just save in them the risks themselves
        self.deterministic_cost_curve = np.zeros(
            shape=(self.ite, self.n_constraints),
            dtype=np.float64
        )
        self.lambda_history = np.zeros(shape=(self.ite, self.n_constraints), dtype=np.float64)
        self.lambda_history[0, :] = deepcopy(self.lambdas)
        self.eta_history = np.zeros(self.ite, dtype=np.float64)
        self.eta_history[0] = deepcopy(self.eta)

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

            # todo: update best theta

            # update best rho
            # fixme we need to use risks to evaluate the constraints!
            self.update_best_rho(current_perf=self.performance_idx[i], risks=self.cost_idx[i, :])

            # Perform Alternate Ascent Descent Algorithm
            if not (i % 2):
                # update the rho vector
                self.update_rho()

                # (if needed) update the eta parameter
                if self.use_eta:
                    self.update_eta()
            else:
                # update lambda
                self.update_lambda()

            # Save the history of the parameters
            self.rho_history[self.time, :] = deepcopy(self.rho[RhoElem.MEAN])
            self.lambda_history[self.time, :] = deepcopy(self.lambdas)
            if self.use_eta:
                self.eta_history[self.time] = deepcopy(self.eta)

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
        self.sample_deterministic_curve()

        return

    def update_rho(self) -> None:
        # Compute the gradient
        m_grad_hat = None
        s_grad_hat = None
        if self.cost_type == "tc":
            # Take the performance of the whole batch: R(tau_i)
            batch_perf = self.performance_idx_theta[self.time, :]

            # Take the costs of the whole batch: C(tau_i)
            batch_cum_cost = self.cost_idx_theta[self.time, :, :]
            batch_cost = np.sum(self.lambdas * batch_cum_cost, axis=1)

            # Combine costs and performances: -R(tau_i) + sum_u(l_u * C_u(tau_i))
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
        elif self.cost_type == "cvar":
            pass
        elif self.cost_type == "mv":
            pass
        elif self.cost_type == "chance":
            pass
        else:
            raise NotImplementedError

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
        grad_hat = None
        if self.cost_type == "tc":
            grad_hat = self.cost_idx[self.time, :] - self.thresholds - self.omega * self.lambdas
        elif self.cost_type == "cvar":
            pass
        elif self.cost_type == "mv":
            pass
        elif self.cost_type == "chance":
            pass
        else:
            raise NotImplementedError

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
        grad_hat = None
        if self.cost_type == "tc":
            return
        elif self.cost_type == "cvar":
            pass
        elif self.cost_type == "mv":
            pass
        elif self.cost_type == "chance":
            pass
        else:
            raise NotImplementedError

        # update the variable
        if self.lr_strategy == "constant":
            self.eta = self.lambdas - self.lr_lambda * grad_hat
        elif self.lr_strategy == "adam":
            self.eta = self.eta + self.eta_adam.compute_gradient(grad_hat)
        else:
            raise NotImplementedError

        return

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

            msg_1 = f"New best RHO: {self.best_rho}"
            msg_2 = f"New best PERFORMANCE: {self.best_performance_rho}"
            msg_3 = f"CONSTRAINT VIOLATION: {violation}"
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
            self, current_perf: float, params: np.ndarray, costs: np.ndarray = None,
            *args, **kwargs
    ) -> None:
        pass

    def save_results(self) -> None:
        """Function saving the results of the training procedure"""
        # Create the dictionary with the useful info
        results = {
            "performance_rho": np.array(self.performance_idx, dtype=float).tolist(),
            "costs_rho": np.array(self.cost_idx, dtype=float).tolist(),
            "performance_thetas_per_rho": np.array(self.performance_idx_theta, dtype=float).tolist(),
            "costs_thetas_per_rho": np.array(self.cost_idx_theta, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "best_rho": np.array(self.best_rho, dtype=float).tolist(),
            "thetas_history": np.array(self.thetas, dtype=float).tolist(),
            "rho_history": np.array(self.rho_history, dtype=float).tolist(),
            "lambda_history": np.array(self.lambda_history, dtype=float).tolist(),
            "eta_history": np.array(self.eta_history, dtype=float).tolist(),
            "deterministic_res": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/cpgpe_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return

    def sample_deterministic_curve(self):
        pass
