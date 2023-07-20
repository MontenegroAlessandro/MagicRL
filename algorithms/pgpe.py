"""
Summary: PGPE implementation
Author: Alessandro Montenegro
Date: 14/7/2023
"""
# Libraries
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor

# Objects
class PGPE:
    """Class implementing PGPE"""
    def __init__(
        self, lr: float = 1e-3, initial_rho: np.array = None, ite: int = 0,
        batch_size: int = 10, episodes_per_theta: int = 10, env: BaseEnv = None,
        policy: BasePolicy = None, 
        data_processor: BaseProcessor = IdentityDataProcessor()
        ) -> None:
        """
        Args:
            lr (float, optional): learning rate. Defaults to 1e-3.
            
            initial_rho (np.array, optional): Initial configuraiton of the 
            hyperpolicy. Each element is assumed to be an array containing
            "[mean, varinace]". Defaults to None.
            
            ite (int, optional): Number of required iterations. Defaults to 0.
            
            batch_size (int, optional): How many theta to sample for each rho
            configuration. Defaults to 10.
            
            episodes_per_theta (int, optional): How many episodes to sample for 
            each theta configuration. Defaults to 10.
            
            env (BaseEnv, optional): The environment in which the agent has to 
            act. Defaults to None.
            
            policy (BasePolicy, optional): The parametric policy to use. Defaults to 
            None.
            
            data_processor (IdentityDataProcessor, optional): the object in 
            charge of transforming the state into a feature vector. Defaults to 
            None.
        """
        # Arguments
        self.lr = lr
        
        assert initial_rho is not None, "[ERROR] No initial hyperpolicy."
        self.rho = initial_rho
        
        self.ite = ite
        self.batch_size = batch_size
        self.episodes_per_theta = episodes_per_theta
        
        assert env is not None, "[ERROR] No env provided."
        self.env = env
        
        assert policy is not None, "[ERROR] No policy provided."
        self.policy = policy
        
        assert data_processor is not None, "[ERROR] No data processor."
        self.data_processor = data_processor
        
        # Other paraemeters
        self.thetas = np.zeros(len(self.rho))
        self.time = 0
        self.performance_idx = np.zeros(ite)
        self.performance_idx_theta = np.zeros((ite, batch_size))
        
    def learn(self):
        """Learning function"""
        for i in range(self.ite):
            for j in range(self.batch_size):
                # Sample theta
                self.sample_theta()
                
                # Collect Trajectories
                perf = self.collect_trajectory()
                self.performance_idx_theta[i, j] = perf
            
            # Update parameters
            self.performance_idx[i] = np.mean(self.performance_idx_theta[i, :])
            self.update_rho()
            
            # Update time counter
            self.time += 1
    
    def update_rho(self):
        """This function modifies the self.rho vector, by updating via the 
        estimated gradient."""
        for id, elem in enumerate(self.rho):
            
            pass
    
    def sample_theta(self):
        """This funciton modifies the self.thetas vector, by sampling parameters
        from the current rho configuration. Each element of rho is assumed to
        be of the form: "[mean, variance]"."""
        for id, elem in self.rho:
            self.thetas[id] = np.random.normal(
                loc=elem[0], scale=np.sqrt(elem[1])
            )
    
    def collect_trajectory(self) -> float:
        """Function collecting a trajectory reward for a particuar theta 
        configuration.
        Returns:
            float: the discounted reward of the trajectory
        """
        # reset the environment
        self.env.reset()
        
        # initialize parameters
        perf = 0
        pol = self.policy(thetas=self.thetas)
        
        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state
            
            # transform the state
            features = self.data_processor.transform(state=state)
            
            # select the action
            a = pol.draw_action(state=features)
            
            # play the action
            _, rew, _ = self.env.step(action=a)
            
            # update the performance index
            perf += (self.env.gamma ** t) * rew
        
        return perf