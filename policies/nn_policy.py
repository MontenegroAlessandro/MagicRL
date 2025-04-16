"""Implementation of a Linear Policy"""
# todo make this modular and fully defined by the user
# Libraries
from policies import LinearGaussianPolicy
from abc import ABC
from policies.utils import NetIdx
import numpy as np
import os

import torch
import torch.nn as nn





def to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).double()
    return x
    
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


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
    
    def zero_grad(self):
        """Zero all parameter gradients"""
        for param in self.parameters():
            param.grad = None

    def tot_parameters(self):
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
class DeepGaussian (LinearGaussianPolicy):
    """
    MLP mapping from states to action distributions
    """
    def __init__(self, dim_state, dim_action, 
                 hidden_neurons=[],  
                 param_init=None,
                 bias=False,
                 activation=torch.tanh,
                 init=torch.nn.init.xavier_uniform_,
                 std_dev=1.0,
                 std_decay=0.0,
                 std_min=1e-6):
        

        super(DeepGaussian, self).__init__(
            parameters = None, #we are using a NN so we want to make sure that we are not using any method in the base class that use parameters
            std_dev = std_dev,
            std_decay = std_decay,
            std_min = std_min,
            dim_state = dim_state,
            dim_action = dim_action
        )

        self.param_init = param_init

        # Mean
        self.mlp = MLPMapping(dim_state, dim_action, 
                             hidden_neurons, 
                             bias, 
                             activation, 
                             init)
        
        self.tot_params = self.mlp.tot_parameters()
        
        if param_init is not None:
            self.mlp.set_from_flat(param_init)


    def calculate_mean(self, state, grad = False):
        if grad:
            return self.mlp(to_torch(state))
        else:
            with torch.no_grad():
                return to_numpy(self.mlp(to_torch(state)))
    
    #ONLY USED BY THE TEST FUNCTION, TO REMOVE IN FINAL VERSION
    def calculate_target_mean(self, state, parameter):
        old_parameter = self.get_parameters()
        self.set_parameters(parameter)
        mean = self.calculate_mean(to_torch(state))
        self.set_parameters(old_parameter)
        return to_numpy(mean)
        

    def compute_log_pi(self, state, action, grad = False):
        """
        Compute the log probability of an action given a state
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float64)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float64)
        
        sigma = torch.tensor(self.std_dev, dtype=torch.float64)

        # Forward pass
        action_mean = self.calculate_mean(state, grad)
        return torch.sum( -0.5 * (((action - action_mean) / sigma) ** 2) - 0.5 * torch.log(2 * torch.pi * sigma ** 2), -1)
    
    def compute_score(self, state, action):
        """
        Compute the score function (gradient of log probability w.r.t. parameters)
        """
        log_prob = self.compute_log_pi(state, action, grad=True)
    
        # Put gradients to zero and compute the gradients
        self.mlp.zero_grad()
        log_prob.backward()
        
        # Extract gradients from model parameters
        grads = torch.nn.utils.parameters_to_vector([p.grad for p in self.mlp.parameters()])
        return np.array(grads.detach(), dtype=np.float64)
    

    def compute_score_all_trajectories(self, states_queue, actions_queue, means):

        scores = np.zeros((states_queue.shape[:-1] + (self.tot_params, )), dtype=np.float64)

        for idx in np.ndindex(states_queue.shape[:-1]):
            scores[idx] = self.compute_score(states_queue[idx], actions_queue[idx])
    
        return scores
    
    
    def set_parameters(self, thetas):
        """
        Set the parameters of the policy
        """
        if isinstance(thetas, np.ndarray):
            thetas = torch.tensor(thetas, dtype=torch.float64)
            
        torch.nn.utils.vector_to_parameters(thetas, self.mlp.parameters())
    
    def get_parameters(self):
        """
        Get the parameters of the policy
        """
        return torch.nn.utils.parameters_to_vector(self.mlp.parameters())
    
