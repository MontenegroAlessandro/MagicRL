"""
Summary: CPGPE implementation
Author: @MontenegroAlessandro
Date: 4/10/2023
"""
# todo -> parallelize the sampling process via joblib
# Libraries
from algorithms.pgpe import PGPE
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import RhoElem, LearnRates
import json, io, os, errno
from tqdm import tqdm
import copy
from overrides import override


# Algorithm implementation
class CPGPE(PGPE):
    """Class implementing CPGPE"""
    def __init__(
            self, lr=None, initial_rho: np.array = None, ite: int = 0,
            batch_size: int = 10, episodes_per_theta: int = 10,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "", verbose: bool = False, natural: bool = False,
            conf_values: list = None, constraints: list = None,
            cost_mask: np.array = None
    ) -> None:
        """
        Args:
            from "lr" to "natural" see PGPE class

            conf_values (list): it is a list of confidence values needed to
            compute the CVaR_{alpha} of the costs. Each element of the list
            must be a float.

            constraints (list): it is a list of thresholds for the CVaR_{alpha}
            of the costs, Each element must be a float.

            cost_mask (np.array (bool)). a mask of booleans for the selection
            of the costs to consider. Default to [... True ...].
        """
        # Super class initialization
        super().__init__(lr=lr, initial_rho=initial_rho, ite=ite,
                         batch_size=batch_size,
                         episodes_per_theta=episodes_per_theta, env=env,
                         policy=policy, data_processor=data_processor,
                         directory=directory, verbose=verbose, natural=natural)

        # CPGPE arguments
        """Learning rates"""
        assert lr is not None, "[ERROR] No learning rates provided."
        assert len(lr) == 3, "[ERROR] Expected 3 learning rates."
        self.lr = lr

        """Constraints Structures"""
        err_msg = "[ERROR] No confidence values for the constraints provided."
        assert len(conf_values) > 0, err_msg
        self.conf_values = conf_values
        self.n_constraints = len(conf_values)

        err_msg = "[ERROR] Number of thresholds != Number of constraints."
        assert len(constraints) == self.n_constraints, err_msg
        self.thresholds = constraints

        if cost_mask is None:
            cost_mask = np.ones(self.n_constraints, dtype=bool)
        err_msg = "[ERROR] Number of cost mask != Number of constraints."
        assert len(cost_mask) == self.n_constraints, err_msg
        self.cost_mask = cost_mask

        """Learning Targets"""
        self.etas = np.zeros(self.n_constraints, dtype=float)
        self.lambdas = np.zeros(self.n_constraints, dtype=float)

        """Useful Structures"""
        # maintain cvar values and costs for each constraint, mean over the
        # iteration
        self.cvar_idx = np.zeros((self.n_constraints, ite), dtype=float)
        self.costs_idx = copy.deepcopy(self.cvar_idx)
        # maintain the cvar and costs value for each theta in each batch (mean)
        self.cvar_idx_theta = np.zeros(
            (self.n_constraints, ite, batch_size),
            dtype=float
        )
        self.costs_idx_theta = copy.deepcopy(self.cvar_idx_theta)
        self.lagrangian = np.zeros(ite, dtype=bool)
        return

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            starting_state = self.env.sample_random_state(n_samples=self.episodes_per_theta)
            for j in range(self.batch_size):
                # Sample theta
                self.sample_theta(index=j)

                # Collect Trajectories
                sample_mean = np.zeros(self.episodes_per_theta, dtype=float)
                cvar_sample_mean = np.zeros((self.n_constraints,
                                             self.episodes_per_theta),
                                            dtype=float)
                cost_sample_mean = copy.deepcopy(cvar_sample_mean)

                for z in range(self.episodes_per_theta):
                    # collect the scores
                    perf_target, perf_costs = self.collect_trajectory(
                        params=self.thetas[j, :],
                        starting_state=starting_state[z]
                    )
                    # update the performance score
                    sample_mean[z] = perf_target
                    # update the inner cvar score
                    k_zeros = np.zeros(self.n_constraints, dtype=float)
                    cvar_sample_mean[:, z] = -np.minimum(k_zeros, perf_costs - self.etas)/self.conf_values
                    cost_sample_mean[:, z] = perf_costs

                # save mean performances
                perf = np.mean(sample_mean)
                self.performance_idx_theta[i, j] = perf

                # save the mean costs
                perf_costs = np.mean(cost_sample_mean, axis=1)
                self.costs_idx_theta[:, i, j] = perf_costs

                # save mean of costs (cvar on a trajectory)
                perf_cvar = np.mean(cvar_sample_mean, axis=1) + self.etas
                self.cvar_idx_theta[:, i, j] = perf_cvar

                # Try to update the best config
                self.update_best_theta(current_perf=perf,
                                       current_costs=perf_cvar,
                                       params=self.thetas[j, :])

            # Update performance J(rho)
            self.performance_idx[i] = np.mean(self.performance_idx_theta[i, :])

            # Update the cvar terms
            self.cvar_idx[:, i] = np.mean(self.cvar_idx_theta[:, i, :], axis=1)

            # Update the lagrangian
            l_cost_term = np.sum(self.lambdas * (-self.cvar_idx[:, i] + self.thresholds))
            self.lagrangian[i] = -self.performance_idx[i] + l_cost_term

            # Update best rho
            self.update_best_rho(current_perf=self.lagrangian[i])

            # Update parameters
            # compute gradients
            eta_grad = self.update_eta(current_ite=i)
            lambda_grad = self.update_lambda(current_ite=i)
            rho_grad_mean, rho_grad_std = self.update_rho()
            # update parameters
            self.etas = self.etas - self.lr[LearnRates.ETA] * eta_grad
            self.lambdas = self.lambdas + self.lr[LearnRates.LAMBDA] * lambda_grad
            self.rho[RhoElem.MEAN] = self.rho[RhoElem.MEAN] - self.lr[
                LearnRates.RHO] * rho_grad_mean
            self.rho[RhoElem.STD] = self.rho[RhoElem.STD] - self.lr[
                LearnRates.RHO] * rho_grad_std

            # Update time counter
            self.time += 1
            if self.verbose:
                print(f"***************END OF BATCH {i}***************")
                print(f"Lagrangian: {self.lagrangian}")
                print(f"rho perf: {self.performance_idx}")
                print(f"theta perf: {self.performance_idx_theta}")
                print(f"rho cvar: {self.cvar_idx}")
                print(f"theta cvar: {self.cvar_idx_theta}")
                print(f"**********************************************\n")

    def collect_trajectory(self, params: np.array,
                           starting_state=None) -> tuple:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta
            configuration.
        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): the starting state for the iterations
        Returns:
            float: the discounted reward of the trajectory
            list (float): the discounted costs from the env (masked as the
            user declares)
        """
        # Reset the environment
        self.env.reset()
        if starting_state is not None:
            self.env.state = copy.deepcopy(starting_state)

        # Initialize parameters
        perf = 0
        perf_costs = np.zeros(self.n_constraints, dtype=float)
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
            _, rew, abs, costs = self.env.step(action=a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew
            perf_costs += (self.env.gamma ** t) * costs

            if self.verbose:
                print("******************************************************")
                print(f"ACTION: {a.radius} - {a.theta}")
                print(f"FEATURES: {features}")
                print(f"REWARD: {rew}")
                print(f"PERFORMANCE: {perf}")
                print(f"COSTS: {perf_costs}")
                print("******************************************************")

            if abs:
                break

        return perf, self.cost_mask * perf_costs

    def update_rho(self) -> tuple:
        """
        Summary:
            this function updates the rho vector via gradient descent
        Args:
            current_ite (int): index of the current iteration of the algorithm.
            It is useful to access rapidly the recording structures.
        Returns:
            The gradient to apply to rho_mu vector.
            The gradient to apply to rho_std vector.
        """
        """build the vector of sigma**2"""
        sigma_squared = np.exp(np.float128(self.rho[RhoElem.STD])) ** 2

        """build the scores vectors"""
        mu_score = (self.thetas - self.rho[RhoElem.MEAN]) / sigma_squared
        sigma_score = ((self.thetas - self.rho[RhoElem.MEAN]) ** 2 - sigma_squared) / sigma_squared
        # remember that we want to update the log(sigma), not the normal sigma

        """compute the gradient pieces"""
        mu_perf_term = - np.mean(mu_score * self.performance_idx_theta[self.time, :])
        sigma_perf_term = - np.mean(sigma_score * self.performance_idx_theta[self.time, :])
        mu_cvar_term = np.zeros(self.dim)
        sigma_cvar_term = np.zeros(self.dim)
        for u in range(self.n_constraints):
            # fixme -> vectorize this part
            mu_cvar_term += self.lambdas[u] * (- np.mean(mu_score * self.cvar_idx_theta[u, self.time, :]))
            sigma_cvar_term += self.lambdas[u] * (- np.mean(sigma_score * self.cvar_idx_theta[u, self.time, :]))

        """compute the gradients"""
        rho_grad_mu = mu_perf_term + mu_cvar_term
        rho_grad_std = sigma_perf_term + sigma_cvar_term
        return rho_grad_mu, rho_grad_std

    def update_eta(self, current_ite: int) -> np.array:
        """
        Summary:
            this function updates the eta vector via gradient descent
        Args:
            current_ite (int): index of the current iteration of the algorithm.
            It is useful to access rapidly the recording structures.
        Returns:
            The gradient to apply to eta vector.
        """
        # inner structures
        ones = np.ones(self.n_constraints, dtype=float)
        in_cvar_term = self.costs_idx[:, current_ite] - self.etas
        cvar_term = np.where(in_cvar_term <= 0, 1, 0)/self.conf_values

        # gradient and update
        eta_grad = - self.lambdas * (cvar_term + ones)
        # self.etas = self.etas - self.lr[LearnRates.ETA] * eta_grad
        return eta_grad

    def update_lambda(self, current_ite: int) -> np.array:
        """
        Summary:
            this function updates the lambda vector via gradient ascent
        Args:
            current_ite (int): index of the current iteration of the algorithm.
            It is useful to access rapidly the recording structures.
        Returns:
            The gradient to apply to lambda vector.
        """
        # gradient and update
        lambda_grad = - self.cvar_idx[:, current_ite] + self.etas + self.thresholds
        # self.lambdas = self.lambdas + self.lr[LearnRates.LAMBDA] * lambda_grad
        return lambda_grad

    def update_best_rho(self, current_perf: float):
        """The best rho is the one having the highest value of lagrangian
        function."""
        super().update_best_rho(current_perf=current_perf)

    @override
    def update_best_theta(self, current_perf: float, current_costs: np.array,
                          params: np.array) -> None:
        """
        Summary:
            this function updates the best theta configuration, defined as the
            one respecting all the constraints and having the highest value of
            performance index.

        Args:
            current_perf (float): the performance index of the considered
            parameter configuration.

            current_costs (np.array): array with length "self.n_constraints"
            having all the mean cvar terms for a single trajectory.

            params (np.array): array storing the parameter configuration that
            is candidate to be the best one.
        """
        """Constraints evaluation"""
        constraints = current_costs - self.thresholds
        violating_ids = np.where(constraints < 0)[0]
        if len(violating_ids) == 0:
            """Check if the theta configuration is the best one"""
            # all the constraints are respected
            super().update_best_theta(current_perf=current_perf, params=params)
        return

    def save_results(self) -> None:
        """Function saving the results of the training procedure"""
        # Create the dictionary with the useful info
        results = {
            "performance_rho": self.performance_idx.tolist(),
            "performance_thetas_per_rho": self.performance_idx_theta.tolist(),
            "costs_rho": self.cvar_idx.tolist(),
            "costs_theta": self.cvar_idx_theta.tolist(),
            "best_theta": self.best_theta.tolist(),
            "best_rho": self.best_rho.tolist(),
            "lagrangian": self.lagrangian
        }

        # Save the json
        name = self.directory + "/cpgpe_results.json"
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
