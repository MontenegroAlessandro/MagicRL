"""Implementation of DPG (deterministic policy gradient):
Compatible Off-Policy Deterministic Actor Critic.
Silver et al., 2015."""

# Libraries
import numpy as np
import torch 
import torch.nn as nn
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import TrajectoryResults, check_directory_and_create
from algorithms.samplers import TrajectorySampler, pg_sampling_worker
from joblib import Parallel, delayed
import json
import io
from tqdm import tqdm
import copy
from adam.adam import Adam


# Algorithm implementation
class DeterministicPG:
    """DPG implementation."""
    def __init__(
        self,
        ite: int = 100,
        directory: str = "",
        det_pol: BasePolicy = None,
        b_pol: BasePolicy = None,
        env: BaseEnv = None,
        value_features: BaseProcessor = None,
        b_pol_features: BaseProcessor = None,
        theta_step: float = 1e-3,
        omega_step: float = 1e-3,
        v_step: float = 1e-3,
        lr_strategy: str = "constant",
        checkpoint_freq: int = 100,
        save_det_curve: bool = False,
        deterministic_sampling_params: dict = None,
        env_seed: int = None,
        update_b_pol: bool = False
    ) -> None:
        # Args checking
        assert ite > 0, "[DPG] ite must be > 0."
        self.ite = ite
        
        assert det_pol is not None, "[DPG] no deterministic policy provided."
        self.det_pol = copy.deepcopy(det_pol)
        
        assert b_pol is not None, "[DPG] no behavioral policy provided."
        self.b_pol = copy.deepcopy(b_pol)
        self.update_b_pol = update_b_pol
        
        assert env is not None, "[DPG] No env provided."
        self.env = copy.deepcopy(env)
        self.env_seed = env_seed
        
        if value_features is None:
            self.value_features = IdentityDataProcessor(dim_feat=self.env.state_dim)
        else:
            self.value_features = copy.deepcopy(value_features)
            
        if b_pol_features is None:
            self.b_pol_features = IdentityDataProcessor(dim_feat=self.env.state_dim)
        else:
            self.b_pol_features = copy.deepcopy(b_pol_features)

        assert theta_step > 0, "[DPG] theta_step must be > 0!"
        self.theta_step = theta_step
        
        assert omega_step > 0, "[DPG] omega_step must be > 0!"
        self.omega_step = omega_step
        
        assert v_step > 0, "[DPG] v_step must be > 0!"
        self.v_step = v_step
        
        assert lr_strategy in ["constant", "adam"], "[DPG] illegal LR_STRATEGY."
        self.lr_strategy = lr_strategy
        
        # V and A approximators
        self.value_function = nn.Sequential(
            nn.Linear(self.value_features.dim_feat, 1, bias=False)
        )
        init_weights = torch.zeros(self.value_features.dim_feat, dtype=torch.float64)
        nn.utils.vector_to_parameters(init_weights, self.value_function.parameters())
        self.advantage_function = nn.Sequential(
            nn.Linear(self.det_pol.tot_params, 1, bias=False)
        )
        init_weights = torch.zeros(self.det_pol.tot_params, dtype=torch.float64)
        nn.utils.vector_to_parameters(init_weights, self.advantage_function.parameters())
        
        # Saving parameters
        self.directory = directory
        check_directory_and_create(self.directory)
        self.checkpoint_freq = checkpoint_freq
        self.theta_history = torch.zeros((self.ite, self.det_pol.tot_params), dtype=torch.float64)
        self.deterministic_curve = None
        
        # Deterministic sampling
        self.save_det_curve = save_det_curve
        self.deterministic_sampling_params = deterministic_sampling_params
        self.deterministic_curve = torch.zeros(self.ite, dtype=torch.float64)
    
    def learn(self):
        # Reset the environment
        state = torch.tensor(self.env.reset(seed=self.env_seed)[0], dtype=torch.float64)
        theta = torch.tensor(self.det_pol.get_parameters(), dtype=torch.float64)
        omega = nn.utils.parameters_to_vector(self.advantage_function.parameters())
        v = nn.utils.parameters_to_vector(self.value_function.parameters())
        
        # Learning Phase
        for i in tqdm(range(self.ite)):
            # Save thetas
            self.theta_history[i,:] = theta.detach().clone()
            
            # Select an action
            raw_action = self.b_pol.draw_action(state)
            action = torch.tensor(raw_action, dtype=torch.float64)
            
            # Collect the result of a step
            next_state, reward, done, _ = self.env.step(raw_action)
            next_state = torch.tensor(next_state, dtype=torch.float64)
            
            # Process the state
            t_value_state = self.value_features.transform(state)
            t_value_next_state = self.value_features.transform(next_state)
            t_b_pol_state = self.b_pol_features.transform(state)
            
            # Update routine
            q_next = 0 if done else self.value_function(t_value_next_state)
            grad_det_pol = self.det_pol.diff(t_value_state)
            omega = nn.utils.parameters_to_vector(self.advantage_function.parameters())
            
            # compute the deltas
            delta = reward + self.env.gamma * q_next - self._Q(state, action)
            # delta_theta = self.theta_step * (grad_det_pol @ (grad_det_pol.T @ omega))
            delta_theta = self.theta_step * (torch.outer(omega, grad_det_pol) @ grad_det_pol)
            delta_omega = self.omega_step * delta * self._nu(state, action)
            delta_v = self.v_step * delta * t_value_state
            
            # updates values
            theta = theta + delta_theta
            self.det_pol.set_parameters(theta)
            if self.update_b_pol:
                self.b_pol.set_parameters(theta)
            
            omega = omega + delta_omega
            nn.utils.vector_to_parameters(omega, self.advantage_function.parameters())
            
            v = v + delta_v
            nn.utils.vector_to_parameters(v, self.value_function.parameters())
            
            # check if data has to be saved
            if i % self.checkpoint_freq == 0:
                self.save_results()
                
            # Change state
            state = next_state.detach().clone()
        
        # Simulate the deterministic curve if needed
        if self.save_det_curve:
            self.sample_deterministic_curve(**self.deterministic_sampling_params)
        
        # Save the results
        self.save_results()
        
        return 
        
    def _Q(self, state: np.array, action: np.array) -> float:
        # Apply feature mapping transformation
        t_state = self.value_features.transform(state)
        
        # Convert to tensors
        t_state = torch.tensor(t_state, dtype=torch.float64, requires_grad=False)
        state = torch.tensor(state, dtype=torch.float64, requires_grad=False)
        action = torch.tensor(action, dtype=torch.float64, requires_grad=False)
        
        # Value computation
        value = self.value_function(t_state).item()
        advantage = self.advantage_function(self._nu(state, action)).item()
        return value + advantage
    
    def _nu(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        # Apply feature mapping transformation
        t_state = self.b_pol_features.transform(state)
        
        # Compute the gradient of the det. pol. w.r.t. the state
        grad_det_pol = torch.tensor(self.det_pol.diff(t_state), dtype=torch.float64)
        
        # Compute the delta and the nu value
        delta = action - torch.tensor(self.det_pol.draw_action(t_state), dtype=torch.float64)
        # return delta @ grad_det_pol
        # return delta * grad_det_pol
        return torch.ravel(
            torch.einsum(
                "i,j->ij", 
                delta, 
                grad_det_pol.view(
                    self.det_pol.dim_action, 
                    self.det_pol.dim_state
                )[0]
            )
        )
    
    def sample_deterministic_curve(
        self,
        ite: int = None, 
        batch: int = None, 
        n_jobs: int = None
    ):
        for i in tqdm(range(0, ite, batch)):
            self.det_pol.set_parameters(thetas=self.theta_history[i, :])
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.det_pol),
                dp=IdentityDataProcessor()
            )
            # build the parallel functions
            delayed_functions = delayed(pg_sampling_worker)

            # parallel computation
            res = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(batch)
            )

            # extract data
            ite_perf = torch.zeros(batch, dtype=torch.float64)
            for j in range(batch):
                ite_perf[j] = res[j][TrajectoryResults.PERF]

            # compute mean
            self.deterministic_curve[i] = torch.mean(ite_perf)
    
    def save_results(self):
        """Function saving the results of the training procedure"""
        # Create the dictionary with the useful info
        results = {
            "thetas_history": self.theta_history.tolist(),
            "deterministic_res": self.deterministic_curve.tolist()
        }

        # Save the json
        name = self.directory + "/dpg_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return