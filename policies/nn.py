"""Implementation of a Linear Policy"""
# todo make this modular and fully defined by the user
# Libraries
from policies import BasePolicy
from abc import ABC
from policies.utils import NetIdx, ActionBoundsIdx
import numpy as np
import torch
import torch.nn as nn


class NeuralNetworkPolicy(BasePolicy, ABC):
    def __init__(
            self, parameters: np.array = None,
            action_bounds: list = None,
            input_size: int = 1,
            output_size: int = 1,
            model: nn.Sequential = None,
            model_desc: dict = None
    ) -> None:
        super().__init__()

        # Attributes with checks
        self.parameters = parameters

        err_msg = "[NNPolicy] too many bounds, 2 or 0 values expected!"
        assert len(action_bounds) == 2 or action_bounds is None, err_msg
        self.action_bounds = action_bounds

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
            self.parameters = np.ones(np.sum(self.params_per_layer))
        self.set_parameters(self.parameters)

    def draw_action(self, state) -> np.array:
        tensor_state = torch.tensor(np.array(state, dtype=np.float64))
        with torch.no_grad():
            raw_action = np.array(self.net.forward(tensor_state))
        action = np.clip(
            raw_action,
            self.action_bounds[ActionBoundsIdx.lb],
            self.action_bounds[ActionBoundsIdx.ub],
            dtype=np.float128
        )
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
        with torch.no_grad():
            for i, param_layer in enumerate(self.net.parameters()):
                if i == 0:
                    batch_params = tensor_param[: self.param_idx[i]]
                elif i == len(self.layers_shape) - 1:
                    batch_params = tensor_param[self.param_idx[i - 1]:]
                else:
                    batch_params = tensor_param[self.param_idx[i - 1]:self.param_idx[i]]
                reshaped_params = torch.reshape(batch_params, self.net_layer_shape[i])
                param_layer.data = nn.parameter.Parameter(reshaped_params)

    def compute_score(self, state, action) -> np.array:
        # todo
        return state
