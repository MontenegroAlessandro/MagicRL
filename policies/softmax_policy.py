"""Implementation of Softmax policies"""
# Libraries
import numpy as np
from policies.base_policy import BasePolicy
from copy import deepcopy

# Tabular Softmax
class TabularSoftmax(BasePolicy):
    def __init__(
        self,
        dim_state: int = 1,
        dim_action: int = 1,
        tot_params: int = 1,
        temperature: float = 1
    ) -> None:
        """
        Summary:
            Initialization.

        Args:
            dim_state (int, optional): dimension of the state space. 
            Defaults to 1.
            
            dim_action (int, optional): dimension of the action space. 
            Defaults to 1.
            
            tot_params (int, optional): dimension of the parameter space. 
            Defaults to 1.
            
            temperature (float, optional): parameter for the softmax policy. 
            Defaults to 1.
        """
        # super-class initialization
        super().__init__()
        
        # parameters
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.tot_params = self.dim_action * self.dim_state
        self.theta = np.zeros((self.dim_state, self.dim_action), dtype=np.float64)
        
        err_msg = f"[TabSoftmax] {temperature} is invalid for the temperature parameter."
        assert temperature > 0, err_msg
        self.temperature = temperature
        
    def draw_action(self, state: int) -> int:
        """
        Summary:
            Funciton that draws an action based on the parameters.

        Args:
            state (int): the index of the state into the tabular representation.
        
        Returns:
            int: index of the action to be selected.
        """
        idx = state
        running_theta = np.zeros(self.dim_action, dtype=np.float64)
        running_theta[:-1] = deepcopy(self.theta[idx, :-1] / self.temperature)
        # norm = 1 + np.sum(np.exp(self.theta[idx, :-1] / self.temperature))
        # probs = np.exp(running_theta) / norm
        
        # get the noise vector for all the actions
        noise = np.random.gumbel(loc=0, scale=1, size=self.dim_action)
        gumbel_scores = running_theta + noise
        return np.argmax(gumbel_scores)
    
    def compute_pi(self, state, action):
        si = state
        ai = action
        running_theta = deepcopy(self.theta[si, ai] / self.temperature)
        norm = 1 + np.sum(np.exp(self.theta[si, :-1] / self.temperature))
        prob = np.exp(running_theta) / norm
        return prob
    
    def set_parameters(self, thetas: np.ndarray, state: int = None, action: int = None):
        """
        Summary: 
            set the parameters of the policy. Normally it sets the parameters 
            for the entire table, else a state or an action index is required.
            Precedence to the state is given.

        Args:
            thetas (np.ndarray): the vector of parameters.
            state (int, optional): state index. Defaults to None.
            action (int, optional): action index. Defaults to None.
        """
        if state is None and action is None:
            self.theta = deepcopy(thetas.reshape((self.dim_state, self.dim_action)))
        elif state is not None:
            self.theta[state,:] = deepcopy(thetas)
        else:
            self.theta[:, action] = deepcopy(thetas)

    def compute_score(self, state: int, action: int) -> np.ndarray:
        """
        Summary: 
            computes the score of the policy as: 
            \nabla_{\theta_state} \log \pi_\theta(action | state) 
                = 1/t (e_action - pi(. | state))

        Args:
            state (int): state index.
            action (int): action index.
        
        Returns:
            np.ndarray: vector of the scores.
        """
        # FIXME: check correctness
        # keep the base vector
        base_vector = np.zeros(self.dim_action - 1, dtype=np.float64)
        # base_vector = np.zeros(self.dim_action, dtype=np.float64)
        if action < self.dim_action - 1:
            base_vector[action] = 1
            
        # compute pi(.|state)
        running_theta = deepcopy(self.theta[state, :-1] / self.temperature)
        norm = 1 + np.sum(np.exp(self.theta[state, :-1] / self.temperature))
        probs = np.exp(running_theta) / norm
        # ! new !
        # complete_probs = np.zeros(self.dim_action, dtype=np.float64)
        # complete_probs[:-1] = deepcopy(probs)
        # complete_probs[-1] = 1 - np.sum(probs)
        
        # score
        score = (1/self.temperature) * (base_vector - probs)
        # score = (1/self.temperature) * (base_vector - complete_probs)
        scores = np.zeros(self.dim_state * self.dim_action).reshape((self.dim_state, self.dim_action))
        scores[state, :-1] = deepcopy(score)
        # scores[state, :] = deepcopy(score)
        return scores.flatten()
    
    def diff(self, state):
        pass

    def reduce_exploration(self):
        pass
    
    def get_parameters(self):
        pass