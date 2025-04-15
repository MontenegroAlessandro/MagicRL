"""Implementation of a Gaussian Policy"""
# Libraries
from policies import BasePolicy
from abc import ABC
import numpy as np
import copy
import time
from collections import defaultdict
import hashlib
from numpy.testing import assert_allclose


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
    
    def calculate_mean(self, state):
        return np.array(self.parameters @ state, dtype=np.float64)
    
    def calculate_mean_cached(self, state):
        state_key = hashlib.md5(state.tobytes()).hexdigest()
        return self.mean_cache[self.current_param_key][state_key]
    

    #expects states in form timesteps x state_dim
    def calculate_mean_full_trajectory(self, states):
        return np.array(states @ self.parameters.T, dtype=np.float64)
    
    def calculate_mean_full_trajectory_cached(self, states, actions):
        timesteps, action_dim = actions.shape
        means = np.zeros((timesteps, action_dim))
        param_key = self.current_param_key #hashlib.md5(self.parameters.tobytes()).hexdigest()

        for step_idx in range(timesteps):
            # Get the state at position [i, j, ...]
            state = states[step_idx]
            # Flatten if needed and compute hash
            flattened_state = state.flatten() if hasattr(state, 'flatten') else state
            state_key = hashlib.md5(flattened_state.tobytes()).hexdigest()
            means[step_idx] = self.mean_cache[param_key][state_key]

        return means
    

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

    def compute_log_pi(self, state, action, cached=False):
        if cached:
            mean = self.calculate_mean_cached(state)
        else:
            mean = self.calculate_mean(state)
        
        fact = 1 / (np.sqrt(2 * np.pi) * self.std_dev)
        log_prob = np.log(fact) - ((action - mean) ** 2) / (2 * (self.var))
        
        return log_prob

    def compute_sum_log_pi(self, states, actions, cached = False):
        
        if cached:
            means = self.calculate_mean_full_trajectory_cached(states, actions) #means is now timesteps x action_dim
        else:
            means = self.calculate_mean_full_trajectory(states) #means is now timesteps x action_dim

        log_fact = -np.log(np.sqrt(2 * np.pi) * self.std_dev)
        log_prob = log_fact - ((actions - means) ** 2) / (2 * self.var)
        
        return np.sum(log_prob)
    


    def compute_sum_all_log_pi(self, states, actions, thetas_queue):
        """Compute sum of log probabilities for multiple parameter sets at once.
        
        Args:
            states: Array of shape (batch_size, timesteps, state_dim)
            actions: Array of shape (batch_size, timesteps, action_dim)
            thetas_queue: target parameter,  of shape (action_dim, state_dim)
            
        Returns:
            log_sums: Array of shape (num_thetas,) containing sum of log probs for each theta
        """
        # Stack all parameters into a single array (num_thetas, action_dim, state_dim)
        thetas = np.stack([theta.reshape(self.dim_action, self.dim_state) 
                          for theta in thetas_queue])
        
        # Compute means for all parameter sets at once
        # (num_thetas, timesteps, action_dim)
        means = np.matmul(states, thetas.transpose(0, 2, 1))


        # While the following works, its way too slow
        """batch_size, timesteps, action_dim = actions.shape
        cached_means = np.zeros((batch_size, timesteps, action_dim))
        param_key = hashlib.md5(thetas_queue[0].tobytes()).hexdigest()

        for batch_idx in range(batch_size):
            for step_idx in range(timesteps):
                # Get the state at position [i, j, ...]
                state = states[batch_idx, step_idx]
                # Flatten if needed and compute hash
                flattened_state = state.flatten() if hasattr(state, 'flatten') else state
                state_key = hashlib.md5(flattened_state.tobytes()).hexdigest()

                try:
                    # Check if the mean is already cached
                    cached_means[batch_idx, step_idx] = self.mean_cache[param_key][state_key]
                except KeyError:
                    cached_means[batch_idx, step_idx] = np.nan
                
        cache_miss_mask = np.isnan(cached_means[:, :, 0]) # misses are True, hence we need to hide what we didnt miss
        expanded_mask = np.broadcast_to(~cache_miss_mask[:, :, np.newaxis], states.shape) # we perform complement such that the mask has True on cache hits
        masked_states = np.ma.array(states, mask=expanded_mask) #cache hits are now hidden

        
        means_test = np.ma.dot(masked_states, thetas_queue.reshape(self.dim_action, self.dim_state).T) # computes masked multiply
        update_mask = np.broadcast_to(cache_miss_mask[:, :, np.newaxis], cached_means.shape)
        cached_means = np.where(update_mask, means_test.data, cached_means) #inserts the new means into the cached means."""
        
        # Broadcasting to compute action deviations
        # (num_thetas, timesteps, action_dim)
        if actions.ndim == 2:
            action_deviations = actions[np.newaxis, :, :] - means


        elif actions.ndim == 3:
            action_deviations = actions - means

        

        # Compute log probabilities
        log_fact = -np.log(np.sqrt(2 * np.pi) * self.std_dev)

        #log probs has dimension batch-size, timesteps, action_dim
        log_probs = log_fact - (action_deviations ** 2) / (2 * self.var)

        return np.sum(log_probs, axis=(1, 2))
    

    def compute_score(self, state, action, cached = False) -> np.array:
        if self.std_dev == 0:
            return super().compute_score(state, action)

        if cached:
            mean = self.calculate_mean_cached(state)
        else:
            mean = self.calculate_mean(state)

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

