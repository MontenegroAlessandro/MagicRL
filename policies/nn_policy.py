"""Implementation of a Linear Policy"""
# todo make this modular and fully defined by the user
# Libraries
from policies import BasePolicy
from abc import ABC
from policies.utils import NetIdx
import numpy as np
import torch
import torch.nn as nn


class NeuralNetworkPolicy(BasePolicy, ABC):
    def __init__(
            self, parameters: np.array = None,
            input_size: int = 1,
            output_size: int = 1,
            model: nn.Sequential = None,
            model_desc: dict = None
    ) -> None:
        super().__init__()

        # Attributes with checks
        self.parameters = parameters

        # Additional attributes
        self.dim_state = input_size
        self.dim_action = output_size

        # Pick the net
        self.net = None
        self.layers_shape = None
        if model is None:
            # Build the default net
            self.net = nn.Sequential(
                nn.Linear(self.dim_state, 32, bias=False),
                nn.Linear(32, 32, bias=False),
                nn.Linear(32, self.dim_action, bias=False)
            )
            self.layers_shape = [
                (self.dim_state, 32),
                (32, 32),
                (32, self.dim_action)
            ]
        else:
            err_msg = "[NNPolicy] model description dictionary is None!"
            assert model_desc is not None, err_msg
            self.net = model
            self.layers_shape = model_desc["layers_shape"]

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
            # initialize the weights to one
            # self.parameters = np.ones(np.sum(self.params_per_layer))
            self.parameters = np.random.normal(0, 1, np.sum(self.params_per_layer))
        self.set_parameters(self.parameters)

    def draw_action(self, state) -> np.array:
        tensor_state = torch.tensor(np.array(state, dtype=np.float64))
        action = np.array(torch.detach(self.net.forward(tensor_state)))
        return action

    def reduce_exploration(self):
        raise NotImplementedError("[NNPolicy] Ops, not implemented yet!")

    def set_parameters(self, thetas) -> None:
        # check on the number of parameters
        err_msg = f"[NNPolicy] Number of parameters {len(thetas)} is different from "
        err_msg += f"{self.tot_params}"
        assert len(thetas) == np.sum(self.params_per_layer), err_msg

        # set the weights
        tensor_param = torch.tensor(np.array(thetas, dtype=np.float64))
        for i, param_layer in enumerate(self.net.parameters()):
            if i == 0:
                batch_params = tensor_param[: self.param_idx[i]]
            elif i == len(self.layers_shape) - 1:
                batch_params = tensor_param[self.param_idx[i - 1]:]
            else:
                batch_params = tensor_param[self.param_idx[i - 1]:self.param_idx[i]]
            reshaped_params = torch.reshape(batch_params, self.net_layer_shape[i])
            param_layer.data = nn.parameter.Parameter(reshaped_params, requires_grad=True)
            
    def get_parameters(self):
        return nn.utils.parameters_to_vector(self.net.parameters())

    def compute_score(self, state, action) -> np.array:
        # todo
        return np.zeros(self.tot_params)
    
    def diff(self, state):
        raise NotImplementedError 


class DeepGaussian(NeuralNetworkPolicy):
    def __init__(
            self, parameters: np.array = None,
            input_size: int = 1,
            output_size: int = 1,
            model: nn.Sequential = None,
            model_desc: dict = None,
            std_dev: float = 1,
            std_decay: float = 0,
            std_min: float = 1e-6
    ) -> None:
        super().__init__(
            parameters=parameters,
            input_size=input_size,
            output_size=output_size,
            model=model,
            model_desc=model_desc
        )
        self.std_dev = std_dev
        self.std_decay = std_decay
        self.std_min = std_min

    def compute_score(self, state, action) -> np.array:
        # Pre-process state and action
        state = torch.tensor(np.array(state, dtype=np.float64))
        action = torch.tensor(np.array(action, dtype=np.float64))
        sigma = torch.tensor(np.array(self.std_dev, dtype=np.float64))

        # Forward pass
        action_mean = self.net.forward(state)
        log_prob = -0.5 * (((action - action_mean) / sigma) ** 2).sum() - 0.5 * torch.log(torch.sqrt(2 * torch.pi * sigma ** 2)) * action.size(0)

        # Put gradients to zero and compute the gradients
        self.net.zero_grad()
        log_prob.backward()

        # set correctly the gradients
        grads = np.zeros(self.tot_params, dtype=np.float64)
        for i, param_layer in enumerate(self.net.parameters()):
            layer_grads = np.array(param_layer.grad, dtype=np.float64)
            if i == 0:
                grads[: self.param_idx[i]] = np.ravel(layer_grads)
            elif i == len(self.layers_shape) - 1:
                grads[self.param_idx[i - 1]:] = np.ravel(layer_grads)
            else:
                grads[self.param_idx[i - 1]:self.param_idx[i]] = np.ravel(layer_grads)

        return grads
    
    def get_parameters(self):
        return super().get_parameters()

    def reduce_exploration(self):
        self.std_dev = np.clip(
            self.std_dev - self.std_decay,
            self.std_min,
            np.inf,
            dtype=np.float64
        )

    def draw_action(self, state) -> np.array:
        means = np.array(super().draw_action(state=state), dtype=np.float64)
        # action = np.array(np.random.normal(means, self.std_dev), dtype=np.float64)
        action = np.array(
            means + self.std_dev * np.random.normal(0, 1, self.dim_action),
            dtype=np.float64
        )
        return action
    
    def diff(self, state):
        raise NotImplementedError 
