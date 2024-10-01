"""CPGPE implementation"""
# todo -> parallelize the sampling process via joblib
# todo -> adam in PGPE and then inherit
# todo -> saving; resume and other stuff in PGPE
# todo -> make CvarPGPE inherit from CPGPE
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
from adam.adam import Adam


# class implementation
class OldCPGPE(PGPE):
    """Class implementing CPGPE"""

    def __init__(
            self, lr=None,
            initial_rho: np.array = None,
            ite: int = 0,
            batch_size: int = 10,
            episodes_per_theta: int = 10,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "", verbose: bool = False, natural: bool = False,
            constraints: list = None,
            cost_mask: np.array = None,
            init_lambda: np.array = None,
            learn_std: bool = True,
            lr_strategy: str = None,
            checkpoint_freq: int = 1,
            resume_from: str = None
    ) -> None:
        """
        todo
        """
        # Super class initialization
        super().__init__(lr=lr, initial_rho=initial_rho, ite=ite,
                         batch_size=batch_size,
                         episodes_per_theta=episodes_per_theta, env=env,
                         policy=policy, data_processor=data_processor,
                         directory=directory, verbose=verbose, natural=natural,
                         checkpoint_freq=checkpoint_freq)
        # CPGPE arguments
        # Learning rates
        assert lr is not None, "[ERROR] No learning rates provided."
        assert len(lr) == 2, "[ERROR] Expected 2 learning rates."
        self.lr = np.array(lr)

        err_msg = "[CPGPE] invalid learning rate update strategy."
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy
        if self.lr_strategy == "adam":
            self.lambda_adam = Adam(step_size=self.lr[LearnRates.LAMBDA], strategy="ascent")

            self.rho_adam = [None, None]
            self.rho_adam[RhoElem.MEAN] = Adam(step_size=self.lr[LearnRates.PARAM],
                                               strategy="descent")
            self.rho_adam[RhoElem.STD] = Adam(step_size=self.lr[LearnRates.PARAM], strategy="descent")

        # Constraint Structures
        err_msg = "[ERROR] Number of thresholds != Number of constraints."
        assert len(constraints) > 0, err_msg
        self.thresholds = np.array(constraints)
        self.n_constraints = len(self.thresholds)

        if cost_mask is None:
            cost_mask = np.ones(self.n_constraints, dtype=bool)
        err_msg = "[ERROR] Number of cost mask != Number of constraints."
        assert len(cost_mask) == self.n_constraints, err_msg
        self.cost_mask = np.array(cost_mask)

        # Learning Targets
        self.learn_std = learn_std
        if init_lambda is None:
            self.lambdas = np.ones(self.n_constraints, dtype=float) * self.cost_mask
        else:
            err_msg = f"[CPGPE] init_lambda does not match {self.n_constraints}"
            assert len(init_lambda) == self.n_constraints, err_msg
            self.lambdas = copy.deepcopy(init_lambda) * self.cost_mask

        # useful structures
        self.costs_idx = np.zeros((self.n_constraints, self.ite), dtype=float)
        self.costs_idx_theta = np.zeros((self.n_constraints, self.ite, self.batch_size))
        self.lagrangian = np.zeros(self.ite, dtype=float)
        self.lambda_history = np.zeros((self.ite, self.n_constraints), dtype=float)
        self.constraints_history = np.zeros((self.ite, self.n_constraints), dtype=float)

        # Resume a previous learning
        if resume_from is not None:
            file = open(resume_from)
            data = json.load(file)
            self.rho = np.array(data["final_rho"])
            self.lambdas = np.array(data["final_lambdas"])
        return

    def learn(self) -> None:
        sampling_args = dict(
            strategy="sphere",
            args=dict(
                n_samples=self.episodes_per_theta,
                density=3,
                radius=2,
                noise=0.1,
                left_lim=0,
                right_lim=np.pi
            )
        )
        """Learning function"""
        for i in tqdm(range(self.ite)):
            starting_state = self.env.sample_state(**sampling_args)
            for j in range(self.batch_size):
                # Sample theta
                self.sample_theta(index=j)

                # Collect Trajectories
                sample_mean = np.zeros(self.episodes_per_theta, dtype=float)
                cost_sample_mean = np.zeros((self.n_constraints, self.episodes_per_theta), dtype=float)

                for z in range(self.episodes_per_theta):
                    # collect the scores
                    perf_target, perf_costs = self.collect_trajectory(
                        params=self.thetas[j, :],
                        starting_state=starting_state[z]
                    )
                    # update the performance score
                    sample_mean[z] = perf_target
                    # update the inner cvar score
                    cost_sample_mean[:, z] = perf_costs

                # save mean performances
                perf = np.mean(sample_mean)
                self.performance_idx_theta[i, j] = perf

                # save the mean costs
                perf_costs = np.mean(cost_sample_mean, axis=1)
                self.costs_idx_theta[:, i, j] = perf_costs

                # Try to update the best config
                self.update_best_theta(current_perf=perf, current_costs=perf_costs, params=self.thetas[j, :])

            # Update performance J(rho)
            self.performance_idx[i] = np.mean(self.performance_idx_theta[i, :])

            # Update the cost idx
            self.costs_idx[:, i] = np.mean(self.costs_idx_theta[:, i, :], axis=1)

            # Update the lagrangian
            l_cost_term = np.sum(self.lambdas * (self.costs_idx[:, i] - self.thresholds))
            self.lagrangian[i] = -self.performance_idx[i] + l_cost_term

            # Update best rho
            self.update_best_rho(current_perf=-self.lagrangian[i])

            """Update history"""
            self.lambda_history[self.time, :] = self.lambdas
            self.constraints_history[self.time, :] = self.costs_idx[:, self.time] - self.thresholds

            """Update parameters"""
            # compute gradients
            lambda_grad = self.update_lambda(current_ite=i) * self.cost_mask
            rho_grad_mean, rho_grad_std = self.update_rho()

            if self.lr_strategy == "constant":
                # update lambda
                self.lambdas = self.lambdas + self.lr[LearnRates.LAMBDA] * lambda_grad
                self.lambdas = np.where(self.lambdas >= 0, self.lambdas, 0)
                # update rho
                self.rho[RhoElem.MEAN] = self.rho[RhoElem.MEAN] - self.lr[
                    LearnRates.PARAM] * rho_grad_mean
                self.rho[RhoElem.STD] = self.rho[RhoElem.STD] - self.lr[
                    LearnRates.PARAM] * rho_grad_std
            elif self.lr_strategy == "adam":
                self.lambdas = self.lambdas + self.lambda_adam.compute_gradient(lambda_grad)
                self.lambdas = np.where(self.lambdas >= 0, self.lambdas, 0)
                self.rho[RhoElem.MEAN] = self.rho[RhoElem.MEAN] - self.rho_adam[
                    RhoElem.MEAN].compute_gradient(rho_grad_mean)
                self.rho[RhoElem.STD] = self.rho[RhoElem.STD] - self.rho_adam[
                    RhoElem.STD].compute_gradient(rho_grad_std)
            else:
                raise NotImplementedError(f"[CPGPE] {self.lr_strategy} not implemented.")

            print(f"******* BATCH {i} *******")
            print(f"LAMBDAs: {self.lambda_history[self.time, :]}")
            print(f"RHO: {self.rho}")
            print(f"CONSTRAINTS: {self.constraints_history[self.time, :]}")
            print(f"BEST PERF Theta: {self.best_performance_theta}")
            print(f"BEST PERF Rho: {self.best_performance_rho}")
            print("****************************")

            # Update time counter
            self.time += 1
            if self.time % self.checkpoint_freq == 0:
                self.save_results()
            if self.verbose:
                print(f"***************END OF BATCH {i}***************")
                print(f"Lagrangian: {self.lagrangian[i]}\n")
                print(f"rho perf: {self.performance_idx}\n")
                print(f"theta perf: {self.performance_idx_theta}\n")
                print(f"rho cvar: {self.costs_idx}\n")
                print(f"theta cvar: {self.costs_idx_theta}\n")
                print(f"**********************************************\n")

        print("==== Learning End ====")
        print(f"RHO: {self.rho}")
        print(f"LAMBDA: {self.lambdas}")
        return

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
        Returns:
            The gradient to apply to rho_mu vector.
            The gradient to apply to rho_std vector.
        """
        """build the vector of sigma**2"""
        sigma_squared = np.exp(np.float64(self.rho[RhoElem.STD])) ** 2
        # sigma = np.exp(np.float64(self.rho[RhoElem.STD]))

        """build the scores vectors"""
        if self.natural:
            mu_score = (self.thetas - self.rho[RhoElem.MEAN])
            sigma_score = ((self.thetas - self.rho[RhoElem.MEAN]) ** 2 - sigma_squared) / (2 * sigma_squared)
        else:
            mu_score = (self.thetas - self.rho[RhoElem.MEAN]) / sigma_squared
            sigma_score = ((self.thetas - self.rho[RhoElem.MEAN]) ** 2 - sigma_squared) / sigma_squared
        # remember that we want to update the log(sigma), not the normal sigma

        """build a utility performance vector"""
        perf = np.ones((self.batch_size, self.dim), dtype=float)
        for i in range(self.batch_size):
            perf[i, :] = self.performance_idx_theta[self.time, i] * perf[i, :]

        """compute the gradient pieces"""
        mu_perf_term = - np.mean(mu_score * perf, axis=0)
        sigma_perf_term = - np.mean(sigma_score * perf, axis=0)
        mu_cost_term = np.zeros(self.dim)
        sigma_cost_term = np.zeros(self.dim)

        # fixme -> vectorize this part
        for u in range(self.n_constraints):
            # utility cost vector
            cost = np.ones((self.batch_size, self.dim), dtype=float)
            for i in range(self.batch_size):
                cost[i, :] = self.costs_idx_theta[u, self.time, i] * cost[i, :]

            mu_cost_term += self.lambdas[u] * np.mean(mu_score * cost, axis=0)
            sigma_cost_term += self.lambdas[u] * np.mean(sigma_score * cost, axis=0)

        """compute the gradients"""
        rho_grad_mu = mu_perf_term + mu_cost_term
        if self.learn_std:
            rho_grad_std = sigma_perf_term + sigma_cost_term
        else:
            rho_grad_std = 0
        return rho_grad_mu, rho_grad_std

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
        lambda_grad = self.costs_idx[:, current_ite] - self.thresholds
        return lambda_grad

    def update_best_rho(self, current_perf: float):
        """The best rho is the one having the highest value of lagrangian
        function."""
        super().update_best_rho(current_perf=current_perf)

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
        violating_ids = np.where(constraints > 0)[0]
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
            "costs_rho": self.costs_idx.tolist(),
            "costs_theta": self.costs_idx_theta.tolist(),
            "best_theta": self.best_theta.tolist(),
            "best_rho": self.best_rho.tolist(),
            "lagrangian": self.lagrangian.tolist(),
            "final_rho": self.rho.tolist(),
            "final_lambdas": self.lambdas.tolist(),
            "lambda_history": self.lambda_history.tolist(),
            "constraint_history": self.constraints_history.tolist()
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