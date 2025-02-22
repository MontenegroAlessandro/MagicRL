"""Policy Gradient Implementation"""
# todo natural
# todo baseline

# Libraries
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import OffPolicyTrajectoryResults, check_directory_and_create, LearnRates, matrix_shift
from algorithms.samplers import TrajectorySampler, off_pg_sampling_worker
from joblib import Parallel, delayed
import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam
import collections
from scipy.special import logsumexp
import logging
import time  # Add this at the top with other imports
logger = logging.getLogger(__name__)

# Class Implementation
class OffPolicyGradient:
    """This Class implements Policy Gradient Algorithms via REINFORCE or GPOMDP."""
    def __init__(
            self, lr: np.array = None,
            lr_strategy: str = "constant",
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
            window_length: int = 5,
            gradient_type: str = "off_pg",
            test: bool = False
    ) -> None:
        """
        Summary:
            Initialization.
        Args:
            lr (np.array, optional): learning rate. Defaults to None.
            
            lr_strategy (str, optional): how to update the learning rate. 
            Choices in "constant" or "adam". Defaults to "constant".
            
            estimator_type (str, optional): how to update the parameters.
            Choices in "REINFORCE" and "GPOMDP". Defaults to "REINFORCE".
            
            initial_theta (np.array, optional): initialization for the parameter
            vector. Defaults to None.
            
            ite (int, optional): how many iteration to run the algorithm.
            Defaults to 100.
            
            batch_size (int, optional): how many trajectories to try for each 
            parameter sampled. Defaults to 1.
            
            env (BaseEnv, optional): which environment to use. Defaults to None.
            
            policy (BasePolicy, optional): which policy to use. Defaults to None.
            
            data_processor (BaseProcessor, optional): which data processor to 
            employ to process the data. Defaults to IdentityDataProcessor().
            
            directory (str, optional): where to save data. Defaults to "".
            
            verbose (bool, optional): whether to log additional information. 
            Defaults to False.
            
            natural (bool, optional): whether to employ the natural gradient. 
            Defaults to False.
            
            checkpoint_freq (int, optional): number of iterations after which 
            results are periodically saved. Defaults to 1.
            
            n_jobs (int, optional): how many trajectories to evaluate in 
            parallel. Defaults to 1.
        """
        # Class' parameter with checks
        err_msg = "[PG] lr must be positive!"
        assert lr[LearnRates.PARAM] > 0, err_msg
        self.lr = lr[LearnRates.PARAM]

        err_msg = "[PG] lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy
        self.test = test

        '''
        err_msg = "[PG] estimator_type not valid!"
        assert estimator_type in ["REINFORCE", "GPOMDP"], err_msg
        self.estimator_type = estimator_type
        '''

        err_msg = "[PG] initial_theta has not been specified!"
        assert initial_theta is not None, err_msg
        self.thetas = np.array(initial_theta)
        self.dim = len(self.thetas)

        err_msg = "[PG] env is None."
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PG] gradient_type not valid!"
        assert gradient_type in ["off_pg", "particle"], err_msg
        self.gradient_type = gradient_type

        err_msg = "[PG] policy is None."
        assert policy is not None, err_msg
        self.policy = policy

        err_msg = "[PG] data processor is None."
        assert data_processor is not None, err_msg
        self.data_processor = data_processor

        check_directory_and_create(dir_name=directory)
        self.directory = directory

        # Other class' parameters
        self.ite = ite
        self.batch_size = batch_size
        self.verbose = verbose
        self.natural = natural
        self.checkpoint_freq = checkpoint_freq
        self.n_jobs = n_jobs
        self.parallel_computation = bool(self.n_jobs != 1)
        self.dim_action = self.env.action_dim
        self.dim_state = self.env.state_dim
        self.window_length = np.min([window_length, self.ite])
        self.window_size = self.window_length * self.batch_size

        # Print algorithm configuration
        logger.info("\n" + "="*50)
        logger.info("Initializing Off-Policy Gradient with parameters:")
        logger.info(f"  - Window Length: {self.window_length}")
        logger.info(f"  - Batch Size: {self.batch_size}")
        logger.info(f"  - Learning Rate: {self.lr}")
        logger.info(f"  - Learning Rate Strategy: {self.lr_strategy}")
        logger.info(f"  - Number of Iterations: {self.ite}")
        logger.info(f"  - Number of Workers: {self.n_jobs}")
        logger.info(f"  - Environment: {self.env.__class__.__name__}")
        logger.info(f"  - Horizon: {self.env.horizon}")
        logger.info(f"  - Test Mode: {self.test}")
        logger.info("="*50 + "\n")

        # Useful structures
        self.theta_history = np.zeros((self.ite, self.dim), dtype=np.float64)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=np.float64)
        self.best_theta = np.zeros(self.dim, dtype=np.float64)
        self.best_performance_theta = -np.inf
        self.sampler = TrajectorySampler(
            env=self.env, pol=self.policy, data_processor=self.data_processor
        )
        self.deterministic_curve = np.zeros(self.ite)

        # init the theta history
        self.theta_history[self.time, :] = copy.deepcopy(self.thetas)
        self.num_updates = 1

        # create the adam optimizers
        self.adam_optimizer = None
        if self.lr_strategy == "adam":
            self.adam_optimizer = Adam(self.lr, strategy="ascent")
        return

    def learn(self) -> None:
        """Learning function"""
        #initialize the queues
        action_queue = collections.deque(maxlen=int(self.window_size))
        state_queue = collections.deque(maxlen=int(self.window_size))
        reward_queue = collections.deque(maxlen=int(self.window_size))

        #initialize thetas queue
        thetas_queue = collections.deque(maxlen=int(self.window_length))
        thetas_queue.append(np.array(self.thetas, dtype=np.float64))

        # initialize product matrix where row i contains the probability product under parameter theta_i
        log_sums = np.zeros((self.window_length, self.window_size), dtype=np.float64)

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
                    starting_state=None,
                    thetas_queue=thetas_queue
                )

                # build the parallel functions
                delayed_functions = delayed(off_pg_sampling_worker)

                # parallel computation
                res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed_functions(**worker_dict) for _ in range(self.batch_size)
                )

            else:
                res = []
                for j in range(self.batch_size):
                    tmp_res = self.sampler.collect_off_policy_trajectory(params=copy.deepcopy(self.thetas), thetas_queue=thetas_queue)
                    res.append(tmp_res)

            # Update performance
            perf_vector = np.zeros(self.batch_size, dtype=np.float64)

            batch_start = (self.num_updates - 1) * self.batch_size
            #for each trajectory in the batch, update the action, state, reward, and score  
            for j in range(self.batch_size):
                perf_vector[j] = res[j][OffPolicyTrajectoryResults.PERF]

                #append the actions and states corresponding to the trajectory
                action_queue.append(np.array(res[j][OffPolicyTrajectoryResults.ActList], dtype=np.float64))
                state_queue.append(np.array(res[j][OffPolicyTrajectoryResults.StateList], dtype=np.float64))
                reward_queue.append(np.array(res[j][OffPolicyTrajectoryResults.RewList], dtype=np.float64))
                log_sums[:self.num_updates, batch_start + j] = np.array(res[j][OffPolicyTrajectoryResults.LogSumList], dtype=np.float64)

            #append the batch of trajectories to the queues

            self.performance_idx[i] = np.mean(perf_vector)
            # Update best theta
            self.update_best_theta(current_perf=self.performance_idx[i])

            # Compute the estimated gradient
            if self.gradient_type == "off_pg":
                estimated_gradient, log_sums = self.calculate_g_off_policy(
                    action_queue=action_queue, state_queue=state_queue,
                    thetas_queue=thetas_queue, reward_queue=reward_queue,
                    log_sums=log_sums
                )
            if self.test:
                test_gradient = self.test_function(
                    action_queue=action_queue, state_queue=state_queue,
                    thetas_queue=thetas_queue, reward_queue=reward_queue
                )
                assert np.abs(np.sum(estimated_gradient) - np.sum(test_gradient)) < 1e-6, "Sum of test gradient does not match sum of estimated gradient"



            # Update parameters
            if self.lr_strategy == "constant":
                self.thetas = self.thetas + self.lr * estimated_gradient
            elif self.lr_strategy == "adam":
                adaptive_lr = self.adam_optimizer.compute_gradient(estimated_gradient)
                self.thetas = self.thetas + adaptive_lr
            else:
                err_msg = f"[PG] {self.lr_strategy} not implemented yet!"
                raise NotImplementedError(err_msg)
            
            #update thetas queue
            thetas_queue.append(np.array(self.thetas, dtype=np.float64))
            self.num_updates = len(thetas_queue)

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

            # save theta history
            self.theta_history[self.time, :] = copy.deepcopy(self.thetas)

            # time update
            self.time += 1

            # reduce the exploration factor of the policy
            self.policy.reduce_exploration()
        self.sample_deterministic_curve()
        return
    

    def calculate_g_off_policy(self, action_queue: collections.deque,
                                            state_queue: collections.deque, 
                                            thetas_queue: collections.deque, 
                                            reward_queue: collections.deque,
                                            log_sums: np.array) -> tuple[np.array, np.array]:
        """
        Summary:
            Calculate the importance sampling ratio.
        Args:
            action_trajectory (collections.deque): the action trajectory.
            state_trajectory (collections.deque): the state trajectory.
            thetas_queue (collections.deque): the thetas trajectory.
            products (np.array): the products matrix.
        Returns:
            np.array: the importance sampling ratio.
        """
        num_trajectories = len(state_queue)
        estimated_gradients = np.zeros((num_trajectories, self.dim), dtype=np.float64)

        #convert the queues to numpy arrays
        reward_array = np.array(reward_queue, dtype=np.float64)
        state_array = np.array(state_queue, dtype=np.float64)
        action_array = np.array(action_queue, dtype=np.float64)
        thetas_array = np.array(thetas_queue, dtype=np.float64)

        #last theta index for the row of the products matrix, it's the last theta from whcih trajectories were sampled.
        theta_idx = self.num_updates - 1

        #for each batch in the window, compute the product of the probabilities
        #products i contains the products of the probabilities under parameter theta_i for all trajectories

        #then i need to recalculate all trajectories with respect to the new parameter
        self.policy.set_parameters(thetas=thetas_queue[theta_idx])
        old_trajectories = num_trajectories - self.batch_size
        log_sums[theta_idx, :(old_trajectories)] = self.policy.compute_sum_all_log_pi(state_array[:old_trajectories], action_array[:old_trajectories], thetas_array[theta_idx].reshape(1, -1))


        if self.num_updates <= 1:
            importance_vector = np.array(1.0, dtype=np.float64)
        else:
            #compute the difference between the log sums of the past trajectories and the log sum of the current trajectory
            log_diff_matrix = np.array(log_sums[:self.num_updates - 1, :num_trajectories] - log_sums[theta_idx, :num_trajectories], dtype=np.float64)

            #computes the log sum exp along the rows of the matrix, fixing columns
            log_sums_stable = logsumexp(log_diff_matrix, axis=0) #compute sum of exponents in log space for stability

            importance_vector = np.array(1.0 / (1.0 + np.exp(log_sums_stable)), dtype=np.float64) #final importance ratio


        importance_vector = importance_vector / self.batch_size

        #compute g, using scores of the past trajectory with respect to the target distribution parameters
        all_scores = self.policy.compute_score_all_trajectories(state_array, action_array)
        all_gradients = self.calculate_all_g(reward_array, all_scores)

        estimated_gradients = importance_vector.reshape(-1, 1) * all_gradients

        if self.num_updates >= self.window_length:
            # In-place operations to modify the original products matrix 
            log_sums = matrix_shift(log_sums, -1, fill_value=0) # First shift up (rows)
            log_sums = matrix_shift(log_sums.T, -self.batch_size, fill_value=0).T # Then shift left (columns)

        return np.sum(estimated_gradients, axis=0), log_sums

    def calculate_g(
            self, reward_trajectory: np.array,
            score_trajectory: np.array
    ) -> np.array:
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

        # Reshape reward_trajectory if it's 1D
        if reward_trajectory.ndim == 1:
            reward_trajectory = reward_trajectory[np.newaxis, :]
            score_trajectory = score_trajectory[np.newaxis, :, :]
        
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** (np.arange(horizon))
        rolling_scores = np.cumsum(score_trajectory, axis=1)
        reward_trajectory = reward_trajectory[:, :, np.newaxis] * rolling_scores
        estimated_gradient = np.mean(
            np.sum(gamma_seq[:, np.newaxis] * reward_trajectory, axis=1),
            axis=0)
        return estimated_gradient

    def calculate_all_g(
            self, reward_trajectories: np.array,
            score_trajectories: np.array
    ) -> np.array:
        """
        Summary:
            Compute GPOMDP gradients for multiple trajectories in parallel.
        Args:
            reward_trajectories (np.array): array of shape (n_trajectories, horizon) containing 
                the rewards obtained in each trajectory.
            
            score_trajectories (np.array): array of shape (n_trajectories, horizon, dim) containing 
                the scores $\\nabla_{\\theta} log \\pi(s_t, a_t)$ for each trajectory.
        Returns:
            np.array: matrix of shape (n_trajectories, dim) where row i contains 
                the estimated gradient for trajectory i.
        """
        gamma = self.env.gamma
        horizon = self.env.horizon
        
        # Ensure inputs are 3D/2D arrays
        if reward_trajectories.ndim == 1:
            reward_trajectories = reward_trajectories[np.newaxis, :]
        if score_trajectories.ndim == 2:
            score_trajectories = score_trajectories[np.newaxis, :, :]
        
        # Create gamma sequence (1, horizon)
        gamma_seq = (gamma * np.ones(horizon, dtype=np.float64)) ** np.arange(horizon)
        
        # Compute cumulative scores (n_trajectories, horizon, dim)
        rolling_scores = np.cumsum(score_trajectories, axis=1)
        
        # Expand rewards for broadcasting (n_trajectories, horizon, 1)
        reward_trajectories = reward_trajectories[:, :, np.newaxis]
        
        # Multiply rewards with rolling scores (n_trajectories, horizon, dim)
        weighted_scores = reward_trajectories * rolling_scores
        
        # Apply gamma discount and sum over time steps
        # gamma_seq[:, np.newaxis] shape: (horizon, 1)
        # Result shape: (n_trajectories, dim)
        estimated_gradients = np.sum(gamma_seq[:, np.newaxis] * weighted_scores, axis=1)
        
        return estimated_gradients
    
    def compute_single_trajectory_scores(self, state_sequence, action_sequence):
        return np.array([self.policy.compute_score(np.array(s), np.array(a)) for s, a in zip(state_sequence, action_sequence)])

    def compute_all_trajectory_log_sum(self, state_queue, action_queue):
        """Compute log sums for all trajectories in parallel"""
        # Use existing n_jobs parameter from class initialization
        log_sums = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self.policy.compute_sum_log_pi)(state_sequence, action_sequence)
            for state_sequence, action_sequence in zip(state_queue, action_queue)
        )
        return np.array(log_sums)


    def test_function(self, action_queue: collections.deque,
                                            state_queue: collections.deque, 
                                            thetas_queue: collections.deque, 
                                            reward_queue: collections.deque) -> np.array:
        """
        Summary: original implementation with products for sanity check
        """
        num_trajectories = len(state_queue)
        num_updates = len(thetas_queue)
        estimated_gradients = np.zeros((num_trajectories, self.dim), dtype=np.float64)
        # initialize product matrix where row i contains the probability product under parameter theta_i
        products = np.zeros((num_updates, num_trajectories), dtype=np.float64)

        #for each batch in the window, compute the product of the probabilities
        #products i contains the products of the probabilities under parameter theta_i for all trajectories
        for i in range(num_updates):
            self.policy.set_parameters(thetas=thetas_queue[i])
            products[i, :] = self.compute_all_trajectory_products(state_queue, action_queue)

        #compute the gradient update
        for trajectory_idx in range(num_trajectories):
            #numerator is product of state/action probabilities using the target distribution
            num = products[-1, trajectory_idx]

            #denomitator is the weigthed sum of the probability product of the trajectory probabilities of all behavioural distributions
            denom = np.sum(products[:, trajectory_idx]) * self.batch_size

            #compute the importance sampling ratio
            importance_sampling_ratio = num / denom

            #compute g, using scores of the past trajectory with respect to the target distribution parameters
            score_trajectory = self.compute_single_trajectory_scores(state_queue[trajectory_idx], action_queue[trajectory_idx])
            g = self.calculate_g(reward_trajectory=reward_queue[trajectory_idx], score_trajectory=score_trajectory)

            estimated_gradients[trajectory_idx] = importance_sampling_ratio * g

        return np.sum(estimated_gradients, axis=0)
    

    def update_best_theta(self, current_perf: np.float64, *args, **kwargs) -> None:
        """
        Summary:
            Updates the best theta configuration.
        Args:
            current_perf (np.float64): teh perforamance obtained by the current 
            theta configuraiton.
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

        # sample
        for i in tqdm(range(self.ite)):
            self.policy.set_parameters(thetas=self.theta_history[i, :])
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.policy),
                dp=IdentityDataProcessor(),
                # params=copy.deepcopy(self.theta_history[i, :]),
                params=None,
                starting_state=None,
                thetas_queue=None,
                deterministic=True

            )
            # build the parallel functions
            delayed_functions = delayed(off_pg_sampling_worker)

            # parallel computation
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(self.batch_size)
            )

            # extract data
            ite_perf = np.zeros(self.batch_size, dtype=np.float64)
            for j in range(self.batch_size):
                ite_perf[j] = res[j][OffPolicyTrajectoryResults.PERF]

            # compute mean
            self.deterministic_curve[i] = np.mean(ite_perf)

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
    def compute_trajectory_product(self, state_sequence, action_sequence):
        """Helper function to compute product for a single trajectory"""
        return np.prod([self.policy.compute_pi(np.array(s), np.array(a)) 
                       for s, a in zip(state_sequence, action_sequence)])

    def compute_all_trajectory_products(self, state_queue, action_queue):
        """Compute probability products for all trajectories in parallel"""
        # Use existing n_jobs parameter from class initialization
        products = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self.compute_trajectory_product)(state_sequence, action_sequence)
            for state_sequence, action_sequence in zip(state_queue, action_queue)
        )
        return np.array(products)
