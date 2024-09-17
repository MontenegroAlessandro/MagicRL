"""
Implementation of Policy Gradient Actor Only (PGAO) in JAX.
"""
import json

# Libraries
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import copy
import jax.numpy as jnp

from envs import Swimmer
from envs.base_env import BaseEnv
from policies.JAX_policies.base_policy_jax import BasePolicyJAX
from policies.JAX_policies.gaussian_policy_jax import LinearGaussianPolicyJAX
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.JAX_algorithms.policy_gradients import PolicyGradients
from algorithms.utils import TrajectoryResults, PolicyGradientAlgorithms
from algorithms.samplers import TrajectorySampler, pg_sampling_worker
from adam.adam import Adam
import io


def worker_process(worker_dict, num_trajectories):
    res = []
    # copy the memory
    for _ in range(num_trajectories):
        res.append(pg_sampling_worker(**worker_dict))
    return res


class PGAO(PolicyGradients):
    def __init__(self,
                 alg: str = PolicyGradientAlgorithms.PG,
                 lr: float = None,
                 ite: int = 100,
                 batch_size: int = 1,
                 env: BaseEnv = None,
                 policy: BasePolicyJAX = None,
                 data_processor: BaseProcessor = IdentityDataProcessor(),
                 natural: bool = False,
                 lr_strategy: str = "constant",
                 checkpoint_freq: int = 1,
                 verbose: bool = False,
                 directory = None,
                 sample_deterministic_curve: bool = False,
                 estimator_type: str = PolicyGradientAlgorithms.REINFORCE,
                 initial_theta: jnp.array = None,
                 n_jobs: int = 1) -> None:
        """
        Summary:
        Initialization of the Policy Gradient Actor Only (PGAO) class.
        Args:
            estimator_type (str): the estimator type, among "REINFORCE" and "GPOMDP". Default is "REINFORCE".

            initial_theta (jnp.array): the initial theta. Default is None.

            n_jobs (int): the number of jobs. Default is 1.
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

        err_msg = self.alg + " estimator_type not in [REINFORCE], [GPOMDP]!"
        assert estimator_type in PolicyGradientAlgorithms.REINFORCE or PolicyGradientAlgorithms.GPOMDP, err_msg
        self.estimator_type = estimator_type

        err_msg = self.alg + " initial_theta has not been specified!"
        assert initial_theta is not None, err_msg
        self.thetas = jnp.array(initial_theta)
        self.dim = len(self.thetas)

        self.n_jobs = n_jobs
        self.parallel_computation = bool(self.n_jobs != 1)
        self.dim_action = self.env.action_dim
        self.dim_state = self.env.state_dim

        # Useful structures
        self.theta_history = jnp.zeros((self.ite, self.dim), dtype=jnp.float64)
        self.performance_idx = jnp.zeros(ite, dtype=jnp.float64)
        self.best_theta = jnp.zeros(self.dim, dtype=jnp.float64)
        self.best_performance_theta = -jnp.inf
        self.sampler = TrajectorySampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor
        )
        self.deterministic_curve = jnp.zeros(self.ite)

        # Init the theta history
        self.theta_history = self.theta_history.at[self.time, :].set(self.thetas)

        # Create the adam optimizers
        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam(self.lr, strategy="ascent")
        return

    def _compute_gradient(self, score_vector, perf_vector, reward_vector) -> jnp.array:
        """
        Summary:
          This function computes the gradient of the objective function.
        """
        if self.estimator_type == PolicyGradientAlgorithms.REINFORCE:
            estimated_gradient = jnp.mean(
                perf_vector[:, jnp.newaxis] * jnp.sum(score_vector, axis=1), axis=0)
        elif self.estimator_type == PolicyGradientAlgorithms.GPOMDP:
            estimated_gradient = self._update_g(reward_trajectory=reward_vector, score_trajectory=score_vector)
        else:
            err_msg = self.estimator_type + " has not been implemented yet!"
            raise NotImplementedError(err_msg)

        # Update the parameters
        if self.lr_strategy == "constant":
            self.thetas = self.thetas.at[:].set(self.thetas + self.lr * estimated_gradient)
        elif self.lr_strategy == "adam":
            adaptive_lr = self.adam_optimizer.compute_gradient(estimated_gradient)
            self.thetas = self.thetas.at[:].set(self.thetas + adaptive_lr)
        else:
            err_msg = self.lr_strategy + " lr_strategy not in 'constant', 'adam'!"
            raise NotImplementedError(err_msg)

        return estimated_gradient

    def _update_g(self, reward_trajectory: jnp.array, score_trajectory: jnp.array) -> jnp.array:
        """
        Summary:
            Update teh gradient estimate accoring to GPOMDP.

        Args:
            reward_trajectory (np.array): array containing the rewards obtained
            in each trajectory of the batch.

            score_trajectory (np.array):  array containing the scores
            $\\nabla_{\\theta} log \\pi(s_t, a_t)$ obtained in each
            trajectory of the batch.

        Returns:
            np.array: the estimated gradient for each parameter.
        """
        gamma = self.env.gamma
        horizon = self.env.horizon
        gamma_seq = (gamma * jnp.ones(horizon, dtype=jnp.float64)) ** (jnp.arange(horizon))
        rolling_scores = jnp.cumsum(score_trajectory, axis=1)
        reward_trajectory = reward_trajectory[:, :, jnp.newaxis] * rolling_scores
        estimated_gradient = jnp.mean(
            jnp.sum(gamma_seq[:, jnp.newaxis] * reward_trajectory, axis=1),
            axis=0)
        return estimated_gradient

    def _update_best_theta(self, current_perf: jnp.float64) -> None:
        """
        Summary:
            Updates the best theta configuration.

        Args:
            current_perf (np.float64): the performance obtained by the current
            theta configuration.
        """
        if self.best_theta is None or self.best_performance_theta <= current_perf:
            self.best_performance_theta = current_perf
            self.best_theta = copy.deepcopy(self.thetas)

            print("#" * 30)
            print("New best parameter configuration found")
            print(f"Performance: {self.best_performance_theta}")
            print(f"Parameter configuration: {self.best_theta}")
            print("#" * 30)
        return

    def _sample_deterministic_curve(self):
        """
        Summary:
            Switch-off the noise and collect the deterministic performance
            associated to the sequence of parameter configuratios seen during
            the learning.
        """
        # make the policy deterministic
        self.policy.std_dev = 0
        self.policy.sigma_noise = 0

        # sample
        for i in tqdm(range(self.ite)):
            self.policy.set_parameters(
                thetas=self.theta_history[i, :])  # It may be necessary to use np.array(self.theta_history[i, :])
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=IdentityDataProcessor(),
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
            ite_perf = jnp.zeros(self.batch_size, dtype=jnp.float64)
            for j in range(self.batch_size):
                ite_perf = ite_perf.at[j].set(res[j][TrajectoryResults.PERF])

            # compute mean
            self.deterministic_curve = self.deterministic_curve.at[i].set(jnp.mean(ite_perf))

    def save_results(self) -> None:
        """Save the results."""
        results = {
            "performance": np.array(self.performance_idx, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist(),
            "thetas_history": np.array(self.theta_history, dtype=float).tolist(),
            "last_theta": np.array(self.thetas, dtype=float).tolist(),
            "best_perf": float(self.best_performance_theta),
            "performance_det": np.array(self.deterministic_curve, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/pg_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return

    def shared_memory_worker(self, worker):
        """
        Summary:
          This function is used to create a shared memory for the policy's Jacobian
          and distribute trajectory collection across multiple jobs.
        """
        worker['pol'].compile_jacobian()

        # Calculate how many trajectories each job should handle
        trajectories_per_job = self.batch_size // self.n_jobs
        extra_trajectories = self.batch_size % self.n_jobs

        # Use joblib to run the worker processes in parallel
        res = Parallel(n_jobs=self.n_jobs)(
            delayed(worker_process)(
                worker,
                trajectories_per_job + (1 if i < extra_trajectories else 0)
            ) for i in range(self.n_jobs)
        )
        res_traj = [item for sublist in res for item in sublist]
        return list(res_traj)

    def learn(self) -> None:
        """
        Summary:
          This function learns the policy.
        """
        policy_dict = dict(
            parameters=np.zeros(self.policy.tot_params, dtype=np.float64),
            dim_state=self.policy.dim_state,
            dim_action=self.policy.dim_action,
            std_dev=np.sqrt(self.policy.std_dev),
            std_decay=0,
            std_min=1e-5,
            multi_linear=self.policy.multi_linear
        )

        for i in tqdm(range(self.ite)):
            if self.parallel_computation:
                # prepare the parameters
                self.policy.set_parameters(np.array(copy.deepcopy(self.thetas), dtype=np.float64))

                worker_dict = dict(
                    env=copy.deepcopy(self.env),
                    pol=copy.deepcopy(LinearGaussianPolicyJAX(**policy_dict)),
                    dp=copy.deepcopy(self.data_processor),
                    params=np.array(self.thetas, dtype=np.float64),
                    starting_state=None,
                    alg=self.alg
                )

                res = self.shared_memory_worker(worker_dict)

            else:
                res = []
                for _ in range(self.batch_size):
                    tmp_res = self.sampler.collect_trajectory(
                        params=copy.deepcopy(self.thetas))  # It may be necessary to use np.array(self.thetas)
                    res.append(tmp_res)

            # Update performance
            perf_vector = jnp.zeros(self.batch_size, dtype=jnp.float64)
            score_vector = jnp.zeros((self.batch_size, self.env.horizon, self.dim), dtype=jnp.float64)
            reward_vector = jnp.zeros((self.batch_size, self.env.horizon), dtype=jnp.float64)

            for j in range(self.batch_size):
                perf_vector = perf_vector.at[j].set(jnp.array(res[j][TrajectoryResults.PERF], dtype=jnp.float64))
                reward_vector = reward_vector.at[j, :].set(
                    jnp.array(res[j][TrajectoryResults.RewList], dtype=jnp.float64))
                score_vector = score_vector.at[j, :, :].set(
                    jnp.array(res[j][TrajectoryResults.ScoreList], dtype=jnp.float64))
            self.performance_idx = self.performance_idx.at[i].set(jnp.mean(perf_vector))

            # Update best rho
            self._update_best_theta(current_perf=self.performance_idx[i])

            # Compute the estimated gradient
            #estimated_gradient = grad(self._objective_function)(score_vector=score_vector, perf_vector=perf_vector, reward_vector=reward_vector)
            estimated_gradient = self._compute_gradient(score_vector=score_vector, perf_vector=perf_vector,
                                                        reward_vector=reward_vector)

            # Log
            if self.verbose:
                print("*" * 30)
                print(f"Step: {self.time}")
                print(f"Mean Performance: {self.performance_idx[self.time - 1]}")
                print(f"Estimated gradient: {estimated_gradient}")
                print(f"Parameter (new) values: {self.thetas}")
                print(f"Best performance so far: {self.best_performance_theta}")
                print(f"Best configuration so far: {self.best_theta}")
                print("*" * 30)

            # Checkpoint
            if self.time % self.checkpoint_freq == 0:
                self.save_results()

            # Save theta history
            self.theta_history = self.theta_history.at[self.time, :].set(copy.deepcopy(self.thetas))

            # Time update
            self.time += 1

            # Reduce the exploration factor of the policy
            self.policy.reduce_exploration()

            # Make a comparison wrt the deterministic performance
            if self.sample_deterministic_curve:
                self._sample_deterministic_curve()


def main():
    ite = 100
    gamma = 1
    env_class = Swimmer
    horizon = 100
    clip = 1
    lr = 0.1
    lr_strategy = "adam"
    n_workers = 6
    batch = 6

    env = env_class(horizon=horizon, gamma=gamma, render=False, clip=bool(clip))
    MULTI_LINEAR = True

    s_dim = env.state_dim
    a_dim = env.action_dim
    tot_params = s_dim * a_dim
    dp = IdentityDataProcessor(dim_feat=env.state_dim)

    policy = LinearGaussianPolicyJAX(
        parameters=np.zeros(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        std_dev=np.sqrt(1),
        std_decay=0,
        std_min=1e-5,
        multi_linear=MULTI_LINEAR
    )

    init_theta = jnp.zeros(tot_params)

    alg_parameters = dict(
        lr=[lr],
        lr_strategy=lr_strategy,
        estimator_type="GPOMDP",
        initial_theta=init_theta,
        ite=ite,
        batch_size=batch,
        env=env,
        policy=policy,
        data_processor=dp,
        verbose=False,
        natural=False,
        checkpoint_freq=100,
        n_jobs=n_workers
    )
    alg = PGAO(**alg_parameters)

    print("START LEARNING")

    alg.learn()
    print(alg.performance_idx)

if __name__ == "__main__":
    main()

