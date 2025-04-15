"""Implementation of a Gaussian Policy"""
# Libraries
from policies import BasePolicy
from abc import ABC
import numpy as np
import copy
import time
from collections import defaultdict
import hashlib


class LinearGaussianPolicy(BasePolicy, ABC):
    """
    Implementation of a Gaussian Policy which is linear in the state.
    Thus, the mean will be: parameters @ state.
    The standard deviation is fixed and is defined by the user.
    """
    def __init__(
            self, parameters: np.array = None,
            std_dev: float = 0.1,
            std_decay: float = 0,
            std_min: float = 1e-4,
            dim_state: int = 1,
            dim_action: int = 1,
            multi_linear: bool = False

    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[GaussPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        err_msg = "[GaussPolicy] standard deviation is negative!"
        assert std_dev > 0, err_msg
        self.std_dev = std_dev
        self.var = std_dev ** 2

        # Additional attributes
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.tot_params = dim_action * dim_state
        self.multi_linear = multi_linear
        self.std_decay = std_decay
        self.std_min = std_min
        

        self.mean_cache = defaultdict(dict)


        return

    def draw_action(self, state) -> float:
        if len(state) != self.dim_state:
            err_msg = "[GaussPolicy] the state has not the same dimension of the parameter vector:"
            err_msg += f"\n{len(state)} vs. {self.dim_state}"
            raise ValueError(err_msg)
        
        #creates unique keys based on hash
        state_key = hashlib.md5(state.tobytes()).hexdigest()

        mean = np.array(self.parameters @ state, dtype=np.float64)

        #each draw action will miss the cache, so we store it for reuse in compute_pi and similar functions
        #there is an extremely small chance that the key is already stored but in that case we just overwrite it as it is the same anyways
        self.mean_cache[self.current_param_key][state_key] = mean

        action = np.array(np.random.normal(mean, self.std_dev), dtype=np.float64)
        return action

    def reduce_exploration(self):
        self.std_dev = np.clip(self.std_dev - self.std_decay, self.std_min, np.inf)

    def set_parameters(self, thetas) -> None:
        self.current_param_key = hashlib.md5(thetas.tobytes()).hexdigest()
        if not self.multi_linear:
            self.parameters = copy.deepcopy(thetas)
        else:
            self.parameters = np.array(np.split(thetas, self.dim_action))
            
    def get_parameters(self):
        return self.parameters
    
    
    def compute_pi(self, state, action):
        # when we compute pi, we can use the cache to speed up the process, we already computed the mean when we drew the action!
        state_key = hashlib.md5(state.tobytes()).hexdigest()

        mean = self.mean_cache[self.current_param_key][state_key]
        fact = 1 / (np.sqrt(2 * np.pi) * self.std_dev)
        prob = fact * np.exp(-((action - mean) ** 2) / (2 * (self.std_dev ** 2)))
        
        return prob 

    def compute_log_pi(self, state, action):
        state_key = hashlib.md5(state.tobytes()).hexdigest()
        mean = self.mean_cache[self.current_param_key][state_key] #since we dont give a specific parameter set, we use the current one
        
        fact = 1 / (np.sqrt(2 * np.pi) * self.std_dev)
        log_prob = np.log(fact) - ((action - mean) ** 2) / (2 * (self.var))
        
        return log_prob

    def compute_sum_log_pi(self, states, actions):
        
        means = np.array(states @ self.parameters.T, dtype=np.float64)
        log_fact = -np.log(np.sqrt(2 * np.pi) * self.std_dev)
        log_prob = log_fact - ((actions - means) ** 2) / (2 * self.var)
        
        return np.sum(log_prob)

    def compute_sum_all_log_pi(self, states, actions, thetas_queue):
        """Compute sum of log probabilities for multiple parameter sets at once.
        
        Args:
            states: Array of shape (timesteps, state_dim)
            actions: Array of shape (timesteps, action_dim)
            thetas_queue: List or array of parameters, each of shape (action_dim, state_dim)
            
        Returns:
            log_sums: Array of shape (num_thetas,) containing sum of log probs for each theta
        """
        # Stack all parameters into a single array (num_thetas, action_dim, state_dim)
        thetas = np.stack([theta.reshape(self.dim_action, self.dim_state) 
                          for theta in thetas_queue])
        
        # Compute means for all parameter sets at once
        # (num_thetas, timesteps, action_dim)
        means = np.matmul(states, thetas.transpose(0, 2, 1))

        param_keys = [hashlib.md5(thetas_queue[i]).hexdigest() for i in range(len(thetas_queue))]

        states_shape = states.shape
        actions_shape = actions.shape

        flattened_states = states.reshape(-1, states_shape[-1])
        state_keys = [hashlib.md5(flattened_states[i].tobytes()).hexdigest() for i in range(len(flattened_states))]

        # dimensions for the cache:
        # (num_thetas, batch_size, timesteps, action_dim)
        cached_means = np.zeros((len(thetas_queue), ) + actions_shape)
        """for i, param_key in enumerate(param_keys):
            for state_key in state_keys:
                try:
                    cached_means[i] = (self.mean_cache[param_key][state_key])
                except KeyError:
                    means_list.append(np.nan)"""


        #means_list = [[self.mean_cache[param_key][state_key] for state_key in state_keys] for param_key in param_keys]
        #mean_list = [[self.mean_cache[param_keys[i]][state_keys[j]]] for j in range(length_state_keys)] for i in range(length_param_keys)]
        
        
        
        #cache_hits = np.array(cache_hits).reshape(-1, actions_shape[-2])
        #means_test = np.array(means_list).reshape(actions_shape)

        #action_deviations_test = actions - means_test
        
        # Broadcasting to compute action deviations
        # (num_thetas, timesteps, action_dim)
        if actions.ndim == 2:
            action_deviations = actions[np.newaxis, :, :] - means


        elif actions.ndim == 3:
            action_deviations = actions - means
            #param_keys = [hashlib.md5(thetas_queue[i]).hexdigest() for i in range(len(thetas_queue))]
            #means_test = np.array([list(self.mean_cache[param_keys[i]].values()) for i in range(len(param_keys))]).reshape(actions.shape)
            #action_deviations_test = actions - means_test

            #log_fact = -np.log(np.sqrt(2 * np.pi) * self.std_dev)
            #log_probs_test = log_fact - (action_deviations_test ** 2) / (2 * self.var)

        

        # Compute log probabilities
        log_fact = -np.log(np.sqrt(2 * np.pi) * self.std_dev)

        #log probs has dimension batch-size, timesteps, action_dim
        log_probs = log_fact - (action_deviations ** 2) / (2 * self.var)
        log_probs_sum = np.sum(log_probs, axis=(1, 2))

        #log_probs_test = log_fact - (action_deviations_test ** 2) / (2 * self.var)
        #log_probs_sum_test = np.sum(log_probs_test, axis=(-2, -1))


        
        # Sum over both timesteps and action dimensions
        return log_probs_sum


    def compute_score(self, state, action) -> np.array:
        if self.std_dev == 0:
            return super().compute_score(state, action)

        state_key = hashlib.md5(state.tobytes()).hexdigest()
        mean = self.mean_cache[self.current_param_key][state_key]


        #state = np.ravel(state)
        action_deviation = action - mean
        if self.multi_linear:
            # state = np.tile(state, self.dim_action).reshape((self.dim_action, self.dim_state))
            action_deviation = action_deviation[:, np.newaxis]
        scores = (action_deviation * state) / (self.std_dev ** 2)
        if self.multi_linear:
            scores = np.ravel(scores)
        return scores

    def compute_score_trajectory(self, states, actions):
        means = states @ self.parameters.T
        action_deviations = actions - means
        scores = (action_deviations[:, :, np.newaxis] * states[:, np.newaxis, :]) / self.var
        return scores.reshape(scores.shape[0], -1)

    def compute_score_all_trajectories(self, states_queue, actions_queue):
        """Compute the score function for multiple trajectories.
        
        Args:
            states_queue: Array of shape (num_trajectories, timesteps, state_dim)
            actions_queue: Array of shape (num_trajectories, timesteps, action_dim)
            
        Returns:
            scores: Array of shape (num_trajectories, timesteps, action_dim * state_dim)
        """
        if states_queue.ndim == 2:
            means = states_queue @ self.parameters.T
            action_deviations = actions_queue - means
            scores = (action_deviations[:, :, np.newaxis] * states_queue[:, np.newaxis, :]) / self.var
            return scores.reshape(scores.shape[0], -1)
        
        # Multiple trajectories version
        means = np.matmul(states_queue, self.parameters.T)  # (num_trajectories, timesteps, action_dim)
        action_deviations = actions_queue - means  # (num_trajectories, timesteps, action_dim)
        
        # Expand dimensions for broadcasting
        action_deviations = action_deviations[:, :, :, np.newaxis]  # (num_trajectories, timesteps, action_dim, 1)
        states_expanded = states_queue[:, :, np.newaxis, :]  # (num_trajectories, timesteps, 1, state_dim)
        
        # Compute scores
        scores = (action_deviations * states_expanded) / self.var  # (num_trajectories, timesteps, action_dim, state_dim)
        
        # Reshape to (num_trajectories, timesteps, action_dim * state_dim)
        return scores.reshape(scores.shape[0], scores.shape[1], -1)
    
    def diff(self, state):
        raise NotImplementedError
    

    def compute_I_alpha (self, states_queue, current_param, past_param, alpha=2):

        num_trajectories, horizon, _ = states_queue.shape
        divergence_sum = 0.0

        for trajectory_idx in range(num_trajectories):
            trajectory_divergence = 1.0

            for timestep in range(horizon):
                state = states_queue[trajectory_idx, timestep]

                current_mean = np.array(current_param.reshape(self.dim_action, self.dim_state)  @ state, dtype=np.float64)
                past_mean = np.array(past_param.reshape(self.dim_action, self.dim_state)  @ state, dtype=np.float64)

                trajectory_divergence *= np.exp( - alpha * (1 - alpha) * np.dot(current_mean - past_mean, current_mean - past_mean) / 2 * self.std_dev**2 )

            divergence_sum += trajectory_divergence

        return divergence_sum / num_trajectories

