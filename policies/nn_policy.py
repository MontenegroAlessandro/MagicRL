"""Implementation of a Linear Policy"""
# todo make this modular and fully defined by the user
# Libraries
from policies import BasePolicy
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


class NeuralNetworkPolicy(BasePolicy, ABC):
    def __init__(
            self, n_states, n_actions, 
            hidden_neurons=[], 
            feature_fun=None, 
            squash_fun=None,
            param_init=None,
            bias=False,
            activation=torch.tanh,
            init=torch.nn.init.xavier_uniform_
    ) -> None:
        super().__init__()

        # Attributes with checks
        self.parameters = param_init

        # Additional attributes
        self.dim_state = n_states
        self.dim_action = n_actions
        self.feature_fun = feature_fun
        self.squash_fun = squash_fun

        # Pick the net
        self.net = None
        self.layers_shape = None
        
        # Build the network based on hidden_neurons
        self.net = nn.Sequential()
        current_dim = self.dim_state
        self.layers_shape = []
        
        # Add hidden layers
        for i, neurons in enumerate(hidden_neurons):
            self.net.add_module(f"linear{i}", nn.Linear(current_dim, neurons, bias=bias))
            self.net.add_module(f"activation{i}", activation)
            self.layers_shape.append((current_dim, neurons))
            current_dim = neurons
            
        # Add output layer
        self.net.add_module(f"linear_out", nn.Linear(current_dim, self.dim_action, bias=bias))
        self.layers_shape.append((current_dim, self.dim_action))

        # Parameter counting
        self.params_per_layer = []
        self.net_layer_shape = []

        for i in range(len(self.layers_shape)):
            n_neurons = self.layers_shape[i][NetIdx.inp] * self.layers_shape[i][NetIdx.out]
            self.params_per_layer.append(n_neurons)
            self.net_layer_shape.append(
                (self.layers_shape[i][NetIdx.out], self.layers_shape[i][NetIdx.inp])
            )

        self.param_idx = np.cumsum(self.params_per_layer)
        self.tot_params = np.sum(self.params_per_layer)

        if self.parameters is None:
            # initialize using Xavier initialization
            self.parameters = np.array([])
            for i in range(len(self.layers_shape)):
                fan_in = self.layers_shape[i][NetIdx.inp]
                fan_out = self.layers_shape[i][NetIdx.out]
                # Xavier formula: sqrt(6 / (fan_in + fan_out))
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                # Uniform distribution between -limit and limit
                layer_params = np.random.uniform(-limit, limit, self.params_per_layer[i])
                self.parameters = np.append(self.parameters, layer_params)
        
        self.set_parameters(self.parameters)

    def forward(self, state):
        """
        Maps state to action
        """
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float64)
            
        if self.feature_fun is not None:
            state = self.feature_fun(state)
        
        action = self.net(state)
        
        if self.squash_fun is not None:
            action = self.squash_fun(action)
            
        return action

    def draw_action(self, state) -> np.array:
        tensor_state = torch.tensor(np.array(state, dtype=np.float64))
        action = np.array(torch.detach(self.forward(tensor_state)))
        return action

    def reduce_exploration(self):
        raise NotImplementedError("[NNPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        # check on the number of parameters
        assert len(thetas) == self.tot_params, "Param mismatch"
        tensor_param = torch.tensor(thetas, dtype=torch.float64)
        torch.nn.utils.vector_to_parameters(tensor_param, self.net.parameters())
            
    def get_parameters(self):
        return torch.nn.utils.parameters_to_vector(self.net.parameters())

    def compute_score(self, state, action) -> np.array:
        # Default implementation
        return np.zeros(self.tot_params)
    
    def diff(self, state):
        raise NotImplementedError


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
    
    def get_flat(self):
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
        
class DeepGaussian(NeuralNetworkPolicy):
    """
    MLP mapping from states to action distributions
    """
    def __init__(self, n_states, n_actions, 
                 hidden_neurons=[], 
                 feature_fun=None, 
                 squash_fun=None,
                 param_init=None,
                 bias=False,
                 activation=torch.tanh,
                 init=torch.nn.init.xavier_uniform_,
                 std_dev=1.0,
                 std_decay=0.0,
                 std_min=1e-6,
                 squash_grads=True):
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.feature_fun = feature_fun
        self.squash_fun = squash_fun
        self.param_init = param_init
        self.squash_grads = squash_grads
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
    
    def forward(self, state):
        """
        Maps state to action mean
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state, dtype=np.float64))
            
        if self.feature_fun is not None:
            state = self.feature_fun(state)
        
        mean_action = self.mlp(state)
        
        if self.squash_fun is not None:
            mean_action = self.squash_fun(mean_action)
            
        return mean_action
    
    def draw_action(self, state):
        """
        Sample an action from the policy
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state, dtype=np.float64))
            
        mean_action = self.forward(state)
        
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
    
    def compute_score(self, state, action):
        """
        Compute the score function (gradient of log probability w.r.t. parameters)
        """
        # Pre-process state and action
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(np.array(state, dtype=np.float64))
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(np.array(action, dtype=np.float64))
        sigma = torch.tensor(np.array(self.std_dev, dtype=np.float64))

        # Forward pass
        action_mean = self.mlp(state)
        log_prob = -0.5 * (((action - action_mean) / sigma) ** 2).sum() - 0.5 * torch.log(torch.sqrt(2 * torch.pi * sigma ** 2)) * action.size(0)

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
        return self.mlp.get_flat()
    
    def reduce_exploration(self):
        """
        Reduce the exploration noise according to the decay rate
        """
        self.std_dev = np.clip(
            self.std_dev - self.std_decay,
            self.std_min,
            np.inf,
            dtype=np.float64
        )