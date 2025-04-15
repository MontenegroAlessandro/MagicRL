"""Implementation of a Linear Policy"""
# todo make this modular and fully defined by the user
# Libraries
from policies import LinearGaussianPolicy
from abc import ABC
from policies.utils import NetIdx
import numpy as np
import torch
import torch.nn as nn


"""Implementation of a Neural Network Policy"""
# Libraries
from policies import BasePolicy
from abc import ABC
from policies.utils import NetIdx
import numpy as np
import torch
import torch.nn as nn


class MLPMapping(nn.Module):
    def __init__(self, d_in, d_out, hidden_neurons, 
                 bias=False, 
                 activation=torch.tanh, 
                 init=nn.init.xavier_uniform_,
                 output_range=None):
        """
        Multi-layer perceptron
        
        d_in: input dimension
        d_out: output dimension
        hidden_neurons: list with number of hidden neurons per layer
        bias: whether to use a bias parameter (default: false)
        """
        super(MLPMapping, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.activation = activation
        
        self.hidden_layers = []
        input_size = self.d_in
        for i, width in enumerate(hidden_neurons):
            layer = nn.Linear(input_size, width, bias)
            init(layer.weight)
            self.add_module("hidden"+str(i), layer)
            self.hidden_layers.append(layer)
            input_size = width
        self.last = nn.Linear(input_size, self.d_out, bias)
        init(self.last.weight)
        self.add_module("last", self.last)
        
        # Output transformation
        if output_range is None:
            self.out = None
        elif type(output_range)==float:
            assert output_range > 0
            self.out = lambda x: torch.tanh(x) * output_range  # [-c, c]
        elif type(output_range)==tuple:
            assert len(output_range)==2
            lower, upper = output_range
            assert upper > lower
            self.out = lambda x: (1 + torch.tanh(x)) * (upper - lower) / 2 + lower
        else:
            raise ValueError("Supported ranges: float (-x, x) or tuple (lower, upper)")
    
    def forward(self, x):
        """Forward pass through the network"""
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        x = self.last(x)
        if self.out is not None:
            x = self.out(x)
        return x
    
    def get_parameters(self):
        """Get flattened parameters"""
        return torch.nn.utils.parameters_to_vector(self.parameters())
    
    def set_parameters(self, params):
        """Set parameters from flattened vector"""
        # Convert to tensor if numpy array
        if isinstance(params, np.ndarray):
            params = torch.tensor(params, dtype=torch.float64)
        torch.nn.utils.vector_to_parameters(params, self.parameters())
    
    def zero_grad(self):
        """Zero all parameter gradients"""
        for param in self.parameters():
            if param.grad is not None:
                param.grad.zero_()

    def tot_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
class DeepGaussian:
    """
    MLP mapping from states to action distributions
    """
    def __init__(self, n_states, n_actions, 
                 hidden_neurons=[], 
                 feature_fun=None, 
                 param_init=None,
                 bias=False,
                 activation=torch.tanh,
                 init=torch.nn.init.xavier_uniform_,
                 std_dev=1.0,
                 std_decay=0.0,
                 std_min=1e-6):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.feature_fun = feature_fun
        self.param_init = param_init
        self.std_dev = std_dev
        self.std_decay = std_decay
        self.std_min = std_min
        
        # Mean
        self.mlp = MLPMapping(n_states, n_actions, 
                             hidden_neurons, 
                             bias, 
                             activation, 
                             init)
        
        self.tot_params = self.mlp.tot_parameters()
        
        if param_init is not None:
            self.mlp.set_from_flat(param_init)
    
    def draw_action(self, state):
        """
        Sample an action from the policy
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state, dtype=np.float64))
            
        mean_action = self.mlp(state)
        
        # Convert to numpy arrays for consistency with reference implementation
        means = np.array(mean_action.detach(), dtype=np.float64)
        action = np.array(
            means + self.std_dev * np.random.normal(0, 1, self.n_actions),
            dtype=np.float64
        )
        
        return action
    
    def diff(self, state):
        """
        Not implemented method (placeholder to match inheritance structure)
        """
        raise NotImplementedError
    
    # Method removed as it's not directly equivalent in the reference implementation

    def compute_log_pi(self, state, action):
        """
        Compute the log probability of an action given a state
        """
        # Pre-process state and action
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float64)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float64)
        sigma = torch.tensor(self.std_dev, dtype=torch.float64)

        # Forward pass
        action_mean = self.mlp(state)
        return -0.5 * (((action - action_mean) / sigma) ** 2).sum() - 0.5 * torch.log(2 * torch.pi * sigma ** 2) * action.size(0)
    
    def compute_score(self, state, action):
        """
        Compute the score function (gradient of log probability w.r.t. parameters)
        """
        log_prob = self.compute_log_pi(state, action)
    
        # Put gradients to zero and compute the gradients
        self.mlp.zero_grad()
        log_prob.backward()
        
        # Extract gradients from model parameters
        grads = torch.nn.utils.parameters_to_vector([p.grad for p in self.mlp.parameters()])
        return np.array(grads.detach(), dtype=np.float64)
    
    def set_parameters(self, thetas):
        """
        Set the parameters of the policy
        """
        self.mlp.set_parameters(thetas)
    
    def get_parameters(self):
        """
        Get the parameters of the policy
        """
        return self.mlp.get_parameters()
    

    def compute_sum_log_pi(self, states, actions):
        """
        Compute the log probabilities of actions given states for a full trajectory
        
        Parameters:
        -----------
        states: array-like or torch.Tensor, shape (batch_size, state_dim)
            Batch of states representing a trajectory
        actions: array-like or torch.Tensor, shape (batch_size, action_dim)
            Batch of actions corresponding to the states
            
        Returns:
        --------
        log_probs: torch.Tensor, shape (batch_size,)
            Log probabilities of each action given its corresponding state
        """
        # Pre-process states and actions
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float64)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float64)
        
        # Ensure we have batch dimension
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            
        sigma = torch.tensor(self.std_dev, dtype=torch.float64)
        
        # Forward pass - this will work on batched inputs
        action_means = self.mlp(states)  # shape: (batch_size, action_dim)
        
        # Calculate log probabilities for each state-action pair
        # For each pair in the batch, compute the log probability
        log_probs = -0.5 * (((actions - action_means) / sigma) ** 2) - 0.5 * torch.log(2 * torch.pi * sigma ** 2)
        return torch.sum(log_probs, dim=(-1, -2)).detach().cpu().numpy()
    
    def compute_sum_all_log_pi(self, states, actions, thetas):
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float64)
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float64)
        if not isinstance(thetas, torch.Tensor):
            thetas = torch.tensor(thetas, dtype=torch.float64)
        
        sums_log_pi = np.zeros((len(thetas), len(states)), dtype=np.float64)

        for i, theta in enumerate(thetas):
            # Set parameters for the current theta
            self.mlp.set_parameters(theta)

            for j, trajectory in enumerate(states):
                # Compute the log probability for the current trajectory
                sums_log_pi[i][j] = self.compute_sum_log_pi(states[j], actions[j])
        
        return sums_log_pi

    
    def compute_score_all_trajectories(self, states_queue, actions_queue):
        """Compute the score function for multiple trajectories.
        
        Args:
            states_queue: Array of shape (num_trajectories, timesteps, state_dim)
            actions_queue: Array of shape (num_trajectories, timesteps, action_dim)
        Returns:
            scores: Array of shape (num_trajectories, timesteps, action_dim * state_dim)
        """
        # Convert inputs to tensors if they aren't already
        if not isinstance(states_queue, torch.Tensor):
            states_queue = torch.tensor(states_queue, dtype=torch.float64)
        if not isinstance(actions_queue, torch.Tensor):
            actions_queue = torch.tensor(actions_queue, dtype=torch.float64)

        if states_queue.ndim == 2: #one trajectory
            scores = np.array([self.compute_score(states_queue[i], actions_queue[i]) for i in range(len(states_queue))], dtype=np.float64)
            return scores
        
        #if not then we have a miltiple trajectories
        num_trajectories = states_queue.shape[0]
        timesteps = states_queue.shape[1]
        
        # Initialize a list to store scores for each trajectory
        all_trajectories_scores = []
        
        # For each trajectory, compute scores for all time steps
        for i in range(num_trajectories):
            trajectory_states = states_queue[i]
            trajectory_actions = actions_queue[i]
            # Computing scores for this trajectory (similar to the single trajectory case)
            trajectory_scores = np.array([self.compute_score(trajectory_states[j], trajectory_actions[j]) 
                                        for j in range(timesteps)], dtype=np.float64)
            all_trajectories_scores.append(trajectory_scores)
        
        # Convert list of trajectories to numpy array
        scores = np.array(all_trajectories_scores, dtype=np.float64)
        
        return scores
    

    def compute_I_alpha (self, states_queue, current_param, past_param, alpha=2):
        if not isinstance(states_queue, torch.Tensor):
            states_queue = torch.tensor(states_queue, dtype=torch.float64)

        num_trajectories, horizon, _ = states_queue.shape
        divergence_sum = 0.0

        for trajectory_idx in range(num_trajectories):
            trajectory_divergence = 1.0

            for timestep in range(horizon):
                state = states_queue[trajectory_idx, timestep]

                self.mlp.set_parameters(current_param)
                current_mean = self.mlp(state)

                self.mlp.set_parameters(past_param)
                past_mean = self.mlp(state)

                trajectory_divergence *= torch.exp( - alpha * (1 - alpha) * torch.dot(current_mean - past_mean, current_mean - past_mean) / 2 * self.std_dev**2 )

            divergence_sum += trajectory_divergence

        return (divergence_sum / num_trajectories).detach().cpu().numpy()


