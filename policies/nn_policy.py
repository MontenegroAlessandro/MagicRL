# Libraries
from policies import LinearGaussianPolicy
import numpy as np

import torch
import torch.nn as nn
from joblib import Parallel, delayed

def to_torch(x):
    if isinstance(x, np.ndarray):
        if not x.flags.writeable:
            x = x.copy()
        return torch.from_numpy(x).double()
    return x
    
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

# def _worker_bh_log_prob(theta, states, actions, policy_config):
#     """
#     Worker function che viene eseguita in un processo separato.
#     Ricostruisce la policy, carica i pesi storici e calcola le log-prob.
#     """
#     worker_config = policy_config.copy()
#     worker_config['n_workers'] = 1 
    
#     pol = DeepGaussian(**worker_config)
#     pol.set_parameters(theta)
    
#     states_torch = to_torch(states)
#     actions_torch = to_torch(actions)
    
#     with torch.no_grad():
#         sigma = torch.tensor(pol.std_dev, dtype=torch.float64)
#         var = sigma ** 2
#         log_fact = -0.5 * torch.log(2 * torch.pi * var)
        
#         means = pol.mlp(states_torch)
        
#         action_deviations = actions_torch - means
#         log_probs = log_fact - (action_deviations ** 2) / (2 * var)
        
#         traj_log_probs = torch.sum(log_probs, dim=(1, 2))
        
#     return traj_log_probs.numpy()

def _worker_bh_log_prob(theta, states, actions, policy_config):
    """
    Worker function che viene eseguita in un processo separato.
    Ricostruisce la policy, carica i pesi storici e calcola le log-prob.
    Gestisce automaticamente sia input (T, D) [singola traj] che (N, T, D) [batch].
    """
    worker_config = policy_config.copy()
    worker_config['n_workers'] = 1 
    
    # 1. Ricostruzione Policy Temporanea
    pol = DeepGaussian(**worker_config)
    pol.set_parameters(theta)
    
    # 2. Conversione a Tensor
    states_torch = to_torch(states)
    actions_torch = to_torch(actions)
    
    # --- GESTIONE DIMENSIONI ---
    # Flag per ricordare se abbiamo artificialmente aggiunto la batch dim
    is_single_trajectory = False
    
    # Se l'input è 2D (T, D), lo facciamo diventare un batch di 1: (1, T, D)
    if states_torch.dim() == 2:
        is_single_trajectory = True
        states_torch = states_torch.unsqueeze(0)   # (1, T, D)
        
    # Idem per le azioni: se (T, D) o (T,), diventano (1, T, D) o (1, T)
    if actions_torch.dim() == states_torch.dim() - 1:
        actions_torch = actions_torch.unsqueeze(0)
    
    with torch.no_grad():
        sigma = torch.tensor(pol.std_dev, dtype=torch.float64)
        var = sigma ** 2
        log_fact = -0.5 * torch.log(2 * torch.pi * var)
        
        # Forward pass: output atteso (N, T, Action_Dim)
        means = pol.mlp(states_torch)
        
        # Gestione broadcasting dimensioni azioni
        # Se actions è (N, T) e means (N, T, A), unsqueeze actions a (N, T, 1)
        if actions_torch.dim() == 2 and means.dim() == 3:
            actions_torch = actions_torch.unsqueeze(-1)
        
        action_deviations = actions_torch - means
        log_probs = log_fact - (action_deviations ** 2) / (2 * var)
        
        # Somma su Time (1) e Action (2) -> Output (N_traj,)
        # Se log_probs è (N, T, A) -> sum(1, 2)
        # Se log_probs è (N, T) -> sum(1)
        sum_dims = (1, 2) if log_probs.dim() == 3 else 1
        traj_log_probs = torch.sum(log_probs, dim=sum_dims)
        
    # Conversione a numpy
    res = traj_log_probs.numpy()

    # --- FIX CRUCIALE PER BROADCASTING ---
    # Se l'input era una singola traiettoria, res ha forma (1,).
    # Dobbiamo restituire lo scalare, così che compute_sum_all_log_pi
    # costruisca un vettore piatto (M,) invece di (M, 1).
    if is_single_trajectory:
        return res[0]
        
    return res


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


class NeuralNetworkPolicy(LinearGaussianPolicy):
    """
    Deterministic MLP mapping from states to actions.
    Keeps the same interface used by the other policies.
    """
    def __init__(self, dim_state, dim_action,
                 hidden_neurons=[],
                 param_init=None,
                 bias=False,
                 activation=torch.tanh,
                 init=torch.nn.init.xavier_uniform_,
                 n_workers=1):

        super(NeuralNetworkPolicy, self).__init__(
            parameters=None,
            std_dev=1.0,
            std_decay=0.0,
            std_min=1e-6,
            dim_state=dim_state,
            dim_action=dim_action
        )

        self.param_init = param_init
        self.n_workers = n_workers

        self.mlp = MLPMapping(dim_state, dim_action,
                              hidden_neurons,
                              bias,
                              activation,
                              init).double()

        self.tot_params = self.mlp.tot_parameters()

        if param_init is not None:
            self.set_parameters(param_init)

    def calculate_mean(self, state, grad=False):
        state_t = to_torch(state)
        if not isinstance(state_t, torch.Tensor):
            state_t = torch.tensor(state_t, dtype=torch.float64)

        if grad:
            return self.mlp(state_t)

        with torch.no_grad():
            return to_numpy(self.mlp(state_t))

    def draw_action(self, state, return_mean=False):
        mean = self.calculate_mean(state, grad=False)

        return mean

    def compute_score(self, state, action):
        return np.zeros(self.tot_params, dtype=np.float64)

    def diff(self, state):
        state_t = to_torch(state)
        if not isinstance(state_t, torch.Tensor):
            state_t = torch.tensor(state_t, dtype=torch.float64)

        self.mlp.zero_grad()
        action = self.mlp(state_t)
        if action.dim() > 1:
            action = torch.sum(action, dim=-1)
        action = torch.sum(action)
        action.backward()

        grads = torch.nn.utils.parameters_to_vector([p.grad for p in self.mlp.parameters()])
        return to_numpy(grads)

    def reduce_exploration(self):
        return

    def set_parameters(self, thetas):
        if isinstance(thetas, np.ndarray):
            thetas = torch.tensor(thetas, dtype=torch.float64)
        torch.nn.utils.vector_to_parameters(thetas, self.mlp.parameters())

    def get_parameters(self):
        return to_numpy(torch.nn.utils.parameters_to_vector(self.mlp.parameters()))
        
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
                 std_min=1e-6,
                 n_workers=1):
        

        super(DeepGaussian, self).__init__(
            parameters = None, #we are using a NN so we want to make sure that we are not using any method in the base class that use parameters
            std_dev = std_dev,
            std_decay = std_decay,
            std_min = std_min,
            dim_state = dim_state,
            dim_action = dim_action
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.param_init = param_init
        self.n_workers = n_workers

        self.policy_config = {
            'dim_state': dim_state,
            'dim_action': dim_action,
            'hidden_neurons': hidden_neurons,
            'bias': bias,
            'activation': activation,
            'init': init,
            'std_dev': std_dev,
            'std_decay': std_decay,
            'std_min': std_min,
            'param_init': None 
        }

        # Mean
        self.mlp = MLPMapping(dim_state, dim_action, 
                             hidden_neurons, 
                             bias, 
                             activation, 
                             init)
        
        self.tot_params = self.mlp.tot_parameters()
        
        if param_init is not None:
            self.mlp.set_from_flat(param_init)

    #scores require grad calculation but we can skip this otherwise
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
    
    #since we don't have a closed form solution for the gradient of the log pi, we need to use the autograd feature of pytorch
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
    
    # given a number of trajectories (num_trajectories, ), or batches of trajectories (batch_trajectory, num_trajectories, )
    # finds the score of each trajectory keeping the same input shape
    def compute_score_all_trajectories(self, states_queue, actions_queue, means):
        
        # Define a function to compute score for a single index
        def compute_score_for_index(idx):
            return idx, self.compute_score(states_queue[idx], actions_queue[idx])
        
        # Get all indices except the last dimension
        indices = list(np.ndindex(states_queue.shape[:-1]))
        
        # Initialize the scores array
        scores = np.zeros((states_queue.shape[:-1] + (self.tot_params, )), dtype=np.float64)
        
        # Parallelize the computation
        results = Parallel(n_jobs=self.n_workers, backend='loky')(delayed(compute_score_for_index)(idx) for idx in indices)
        
        # Populate the scores array with the results
        for idx, score in results:
            scores[idx] = score
        
        return scores
    
    def compute_sum_all_log_pi(self, states, actions, thetas_queue):
        """
        Compute sum of log probabilities for multiple parameter sets (BH support).
        Uses Joblib to parallelize over CPU cores.
        """
        num_thetas = thetas_queue.shape[0]

        if self.n_workers == 1:
            results = []
            for i in range(num_thetas):
                res = _worker_bh_log_prob(thetas_queue[i], states, actions, self.policy_config)
                results.append(res)
            return np.array(results)
        else:
            results = Parallel(n_jobs=self.n_workers, backend='loky')(
                delayed(_worker_bh_log_prob)(
                    thetas_queue[i], 
                    states, 
                    actions, 
                    self.policy_config
                ) 
                for i in range(num_thetas)
            )
            
            return np.array(results)
    
    def set_parameters(self, thetas):
        """
        Set the parameters of the policy
        """
        if isinstance(thetas, np.ndarray):
            thetas = torch.tensor(thetas, dtype=torch.float64)
            
        torch.nn.utils.vector_to_parameters(thetas, self.mlp.parameters())
        
    def get_parameters(self):
        return to_numpy(torch.nn.utils.parameters_to_vector(self.mlp.parameters()))
