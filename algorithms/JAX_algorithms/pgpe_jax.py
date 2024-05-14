"""
PGPE implementation using JAX.
"""
import io
import json

# libraries
from algorithms.JAX_algorithms.policy_gradients import PolicyGradients
from algorithms.utils import LearnRates, ParamSamplerResults, PolicyGradientAlgorithms
from adam.adam import Adam
from data_processors import IdentityDataProcessor
from algorithms.samplers import *
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
from jax import grad, jit, vmap, jacfwd, jacrev


class PGPE_JAX(PolicyGradients):
    def __init__(self,
                 alg: str = PolicyGradientAlgorithms.PG,
                 lr: jnp.array = None,
                 ite: int = 100,
                 batch_size: int = 10,
                 env: BaseEnv = None,
                 policy: BasePolicy = None,
                 data_processor: BaseProcessor = IdentityDataProcessor(),
                 natural: bool = False,
                 lr_strategy: str = "constant",
                 checkpoint_freq: int = 1,
                 verbose: bool = False,
                 sample_deterministic_curve: bool = False,
                 directory: str = "",
                 initial_rho: jnp.array = None,
                 episodes_per_theta: int = 10,
                 learn_std: bool = False,
                 std_decay: float = 0,
                 std_min: float = 1e-4,
                 n_jobs_param: int = 1,
                 n_jobs_traj: int = 1
                 ) -> None:
        """
        Initialization of the Policy Gradients with Parameter-Based Exploration (PGPE) class with the JAX implementation
        Args:
             initial_rho (np.array, optional): Initial configuration of the
            hyperpolicy. Each element is assumed to be an array containing
            "[mean, log(std_dev)]". Defaults to None.

            episodes_per_theta (int, optional): How many episodes to sample for
            each theta configuration. Defaults to 10.

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
        super().__init__(alg=alg,
                         lr=lr,
                         ite=ite,
                         batch_size=batch_size,
                         env=env,
                         policy=policy,
                         data_processor=data_processor,
                         natural=natural,
                         lr_strategy=lr_strategy,
                         checkpoint_freq=checkpoint_freq,
                         verbose=verbose,
                         sample_deterministic_curve=sample_deterministic_curve,
                         directory=directory)

        err_msg = "[" + alg + "]" + " [ERROR] No initial_rho provided"
        assert lr is not None, err_msg
        self.lr = lr[LearnRates.RHO]

        err_msg = "[" + alg + "]" + " [ERROR] No initial hyperpolicy."
        assert initial_rho is not None, err_msg
        self.rho = initial_rho
        self.dim = len(self.rho[RhoElem.MEAN])

        if self.lr_strategy == "adam":
            self.rho_adam = [None, None]
            self.rho_adam[RhoElem.MEAN] = Adam(step_size=self.lr, strategy="ascent")
            self.rho_adam[RhoElem.STD] = Adam(step_size=self.lr, strategy="ascent")

        self.episodes_per_theta = episodes_per_theta
        self.learn_std = learn_std
        self.std_decay = std_decay
        self.std_min = std_min
        self.n_jobs_param = n_jobs_param
        self.n_jobs_traj = n_jobs_traj

        # Additional parameters
        if len(self.rho[RhoElem.STD]) != self.dim:
            raise ValueError("[PGPE] different size in RHO for µ and σ.")
        self.thetas = jnp.zeros((self.batch_size, self.dim), dtype=jnp.float64)
        self.performance_idx_theta = jnp.zeros((ite, batch_size), dtype=jnp.float64)
        self.parallel_computation_param = bool(self.n_jobs_param != 1)
        self.parallel_computation_traj = bool(self.n_jobs_traj != 1)
        self.sampler = ParameterSampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor,
            episodes_per_theta=self.episodes_per_theta, n_jobs=self.n_jobs_traj
        )

        # Saving parameters
        self.best_theta = jnp.zeros(self.dim, dtype=jnp.float64)
        self.best_rho = self.rho
        self.best_performance_theta = -np.inf
        self.best_performance_rho = -np.inf
        self.checkpoint_freq = checkpoint_freq
        self.deterministic_curve = jnp.zeros(self.ite, dtype=jnp.float64)

        self.rho_history = jnp.zeros((ite, self.dim), dtype=jnp.float64)
        self.rho_history = self.rho_history.at[self.time, :].set(copy.deepcopy(self.rho[RhoElem.MEAN]))

        # compile the function to call the jacobian
        self.jacobian_mean = jit(jacfwd(self._objective_function, argnums=0))
        self.jacobian_std = jit(jacfwd(self._objective_function, argnums=1))

        return

    def _update_best_theta(self, current_perf: float, params: jnp.array) -> None:
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
            print("*" * 30)
            print(f"New best THETA: {self.best_theta}")
            print(f"New best PERFORMANCE: {self.best_performance_theta}")
            print("*" * 30)

            # Save the best theta configuration
            if self.directory != "":
                file_name = self.directory + "/best_theta"

            else:
                file_name = "best_theta"
            np.save(file_name, self.best_theta)
        return

    def _update_best_rho(self, current_perf: float):
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
            if self.directory != "":
                file_name = self.directory + "/best_rho"
            else:
                file_name = "best_rho"
            np.save(file_name, self.best_rho)
        return

    def _objective_function(self, means, stds, thetas) -> jnp.array:
        """
        Summary:
        This function computes the objective for the PGPE hyperpolicy
        """
        # return logarithm of the distribution of the hyperpolicy, which is gaussian
        obj = ((thetas - means) ** 2) / (2 * (stds ** 2))

        # todo implement the natural gradient

        return obj

    def _update_rho(self) -> None:
        """
        Summary:
            This function updates the hyperpolicy parameters using the
            performance of the current batch of thetas
        """

        # Take the performance of the whole batch
        batch_perf = self.performance_idx_theta[self.time, :]

        # Take the means and the sigmas
        means = self.rho[RhoElem.MEAN, :]
        stds = jnp.float64(jnp.exp(self.rho[RhoElem.STD, :]))

        # Compute gradients using JAX grad function

        # (-1) since jacfwd is inverting the sign of the gradient
        grad_means = batch_perf[:, jnp.newaxis] * vmap(
            lambda theta: jnp.diag((-1) * self.jacobian_mean(means, stds, theta)))(self.thetas)

        if self.learn_std:
            grad_stds = batch_perf[:, jnp.newaxis] * vmap(
                lambda theta: jnp.diag((-1) * self.jacobian_std(means, stds, theta)))(self.thetas)
        else:
            grad_stds = jnp.zeros(len(grad_means), dtype=jnp.float64)

        # update rho
        if self.lr_strategy == "constant":
            self.rho = self.rho.at[RhoElem.MEAN, :].set(self.rho[RhoElem.MEAN, :] + self.lr * jnp.mean(grad_means, axis=0))
            # update sigma if it is the case
            if self.learn_std:
                self.rho = self.rho.at[RhoElem.STD, :].set(self.rho[RhoElem.STD, :] + self.lr * jnp.mean(grad_stds, axis=0))
        elif self.lr_strategy == "adam":
            adaptive_lr_m = self.rho_adam[RhoElem.MEAN].compute_gradient(np.mean(grad_means, axis=0))
            adaptive_lr_m = jnp.array(adaptive_lr_m, dtype=jnp.float64)
            self.rho = self.rho.at[RhoElem.MEAN, :].set(self.rho[RhoElem.MEAN, :] + adaptive_lr_m)
            # update sigma if it is the case
            if self.learn_std:
                adaptive_lr_s = self.rho_adam[RhoElem.STD].compute_gradient(np.mean(grad_stds, axis=0))
                adaptive_lr_s = jnp.array(adaptive_lr_s, dtype=jnp.float64)
                self.rho = self.rho.at[RhoElem.STD, :].set(self.rho[RhoElem.STD, :] + adaptive_lr_s)
        else:
            raise NotImplementedError(f"[{str(self.alg)}] Ops, not implemented yet!")

        return

    def _sample_deterministic_curve(self):
        """
        Summary:
            This sample computes the deterministic curve associated with the
            sequence of hyperparameter configuration seen during the learning.
        """
        for i in tqdm(range(self.ite)):
            self.policy.set_parameters(thetas=np.array(self.rho_history[i, :], dtype=np.float64))
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
            res = jnp.array(Parallel(n_jobs=self.n_jobs_param, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(self.batch_size)), dtype=jnp.float64)

            # extract data
            ite_perf = jnp.zeros(self.batch_size, dtype=jnp.float64)
            for j in range(self.batch_size):
                ite_perf = ite_perf.at[j].set(res[j][TrajectoryResults.PERF])

            # compute mean
            self.deterministic_curve = self.deterministic_curve.at[i].set(jnp.mean(ite_perf))

    def _save_results(self) -> None:
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
        name = self.directory + "/pgpe_jax_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
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
                    params=copy.deepcopy(np.array(self.rho, dtype=np.float64)),
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
                    res.append(self.sampler.collect_trajectories(params=copy.deepcopy(np.array(self.rho, dtype=np.float64))))

            # post-processing of results
            performance_res = jnp.zeros(self.batch_size, dtype=jnp.float64)
            for z in range(self.batch_size):
                self.thetas = self.thetas.at[z, :].set(jnp.array(res[z][ParamSamplerResults.THETA], dtype=jnp.float64))
                performance_res = performance_res.at[z].set(jnp.mean(jnp.array(res[z][ParamSamplerResults.PERF], dtype=jnp.float64)))
            self.performance_idx_theta = self.performance_idx_theta.at[i, :].set(performance_res)

            # try to update the best theta
            max_batch_perf = jnp.max(performance_res)
            best_theta_batch_index = jnp.where(performance_res == max_batch_perf)[0]

            self._update_best_theta(
                current_perf=np.float64(max_batch_perf), params=self.thetas[best_theta_batch_index, :]
            )

            # Update performance
            self.performance_idx = self.performance_idx.at[i].set(jnp.mean(self.performance_idx_theta[i, :]))

            # Update best rho
            self._update_best_rho(current_perf=np.float64(self.performance_idx[i]))

            # Update parameters
            self._update_rho()

            # save the current rho configuration
            self.rho_history = self.rho_history.at[self.time, :].set(copy.deepcopy(self.rho[RhoElem.MEAN]))

            # Update time counter
            self.time += 1
            if self.verbose:
                print(f"rho perf: {self.performance_idx}")
                print(f"theta perf: {self.performance_idx_theta}")
            if self.time % self.checkpoint_freq == 0:
                self._save_results()

            # std_decay
            if not self.learn_std:
                std = jnp.float64(jnp.exp(self.rho[RhoElem.STD]))
                std = jnp.clip(std - self.std_decay, self.std_min, jnp.inf)
                self.rho = self.rho.at[RhoElem.STD, :].set(jnp.log(std))

        # Sample the deterministic curve
        if self.sample_deterministic_curve:
            self._sample_deterministic_curve()

        return
