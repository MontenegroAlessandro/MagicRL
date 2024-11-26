"""Implementation of a Linear Policy"""
# Libraries
from policies import BasePolicy
from abc import ABC
import numpy as np
import copy
import torch
import torch.nn as nn


class OldLinearPolicy(BasePolicy, ABC):
    """
    Implementation of a Linear Policy in the state vector.
    Thus, the action will be: parameters @ state.
    """
    def __init__(
            self, parameters: np.array = None,
            dim_state: int = 1,
            dim_action: int = 1,
            multi_linear: bool = False
    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes with checks
        err_msg = "[LinPolicy] parameters is None!"
        assert parameters is not None, err_msg
        self.parameters = parameters

        # Additional attributes
        self.multi_linear = multi_linear
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.tot_params = dim_state * dim_action

        return

    def draw_action(self, state) -> float:
        if len(state) != self.dim_state:
            err_msg = f"[LinPolicy] the state has not the same dimension of the parameter vector:"
            err_msg += f"{len(state)} vs. {self.dim_state}"
            raise ValueError(err_msg)
        action = self.parameters @ state
        return action

    def reduce_exploration(self):
        raise NotImplementedError("[LinPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        if not self.multi_linear:
            self.parameters = copy.deepcopy(thetas)
        else:
            self.parameters = np.array(np.split(thetas, self.dim_action))
            
    def get_parameters(self):
        return self.parameters

    def compute_score(self, state, action) -> np.array:
        if self.multi_linear:
            state = np.tile(state, self.dim_action)
        return state
    
    def diff(self, state):
        raise NotImplementedError


class LinearPolicy(BasePolicy, ABC):
    def __init__(
        self, parameters: np.array = None,
        dim_state: int = 1,
        dim_action: int = 1,
        sigma_noise: float = 0,
        sigma_decay: float = 0,
        sigma_min: float = 1e-5
    ) -> None:
        # Superclass initialization
        super().__init__()

        # Attributes
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.tot_params = dim_state * dim_action
        self.sigma_noise = torch.tensor(sigma_noise, dtype=torch.float64, requires_grad=False)
        self.multi_linear = bool(self.dim_action > 1)
        self.sigma_decay = sigma_decay
        
        assert sigma_min > 0, "[LinPol] sigma_min cannot be < 0 !"
        self.sigma_min = sigma_min
        
        # Parameters initialization
        if parameters is None:
            self.parameters = torch.zeros(self.tot_params, dtype=torch.float64, requires_grad=True)
        else:
            self.parameters = copy.deepcopy(torch.tensor(parameters, dtype=torch.float64, requires_grad=True))
        
        # Model instantiation
        self.pol = nn.Sequential(
            nn.Linear(self.dim_state, self.dim_action, bias=False)
        )
        nn.utils.vector_to_parameters(self.parameters, self.pol.parameters())
        
    def draw_action(self, state) -> np.array:
        state = torch.tensor(state, dtype=torch.float64, requires_grad=False)
        self.sigma_noise = torch.tensor(self.sigma_noise, dtype=torch.float64, requires_grad=False)
        
        noise = torch.normal(mean=0, std=self.sigma_noise, size=(self.dim_action,))
        action = self.pol(state) + noise
        return action.detach().numpy()

    def reduce_exploration(self):
        self.sigma_noise = torch.clamp(
            self.sigma_noise - self.sigma_decay, 
            self.sigma_min, 
            torch.inf
        )

    def set_parameters(self, thetas) -> None:
        thetas = torch.tensor(thetas, dtype=torch.float64, requires_grad=True)
        
        nn.utils.vector_to_parameters(
            thetas, 
            self.pol.parameters()
        )
        
    def get_parameters(self):
        return nn.utils.parameters_to_vector(self.pol.parameters())

    def compute_score(self, state, action) -> np.array:
        state = torch.tensor(state, dtype=torch.float64, requires_grad=False)
        action = torch.tensor(action, dtype=torch.float64, requires_grad=False)
        
        if self.sigma_noise == 0:
            state = state.numpy()
            if self.multi_linear:
                state = np.tile(state, self.dim_action)
            return state
        else:
            action_deviation = action - self.pol(state)
            log_prob = -0.5 * ((action_deviation / self.sigma_noise) ** 2).sum() 
            log_prob -= 0.5 * torch.log(torch.sqrt(2 * torch.pi * self.sigma_noise ** 2)) * action.size(0)

            # Put gradients to zero
            self.pol.zero_grad()
            
            # Gradients Extraction
            gradients = torch.autograd.grad(
                log_prob, 
                self.pol.parameters(), 
                create_graph=True
            )
            grads = torch.nn.utils.parameters_to_vector(gradients)
            
            return grads.detach().numpy()
        
    def diff(self, state):
        if self.sigma_noise > 0:
            raise NotImplementedError("[LinPol] no diff for stochastic policy.")
        else:
            return state.repeat(self.dim_action)
