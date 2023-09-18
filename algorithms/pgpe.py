"""
Summary: PGPE implementation
Author: @MontenegroAlessandro
Date: 14/7/2023
"""
# Libraries
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import RhoElem
import json, io, os, errno
from tqdm import tqdm


# from adam import Adam
# TODO: implement adam please
# TODO: implement a parallel sampler

# Objects
class PGPE:
    """Class implementing PGPE"""

    def __init__(
            self, lr: float = 1e-3, initial_rho: np.array = None, ite: int = 0,
            batch_size: int = 10, episodes_per_theta: int = 10,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "", verbose: bool = False
    ) -> None:
        """
        Args:
            lr (float, optional): learning rate. Defaults to 1e-3.
            
            initial_rho (np.array, optional): Initial configuraiton of the 
            hyperpolicy. Each element is assumed to be an array containing
            "[mean, log(varinace)]". Defaults to None.
            
            ite (int, optional): Number of required iterations. Defaults to 0.
            
            batch_size (int, optional): How many theta to sample for each rho
            configuration. Defaults to 10.
            
            episodes_per_theta (int, optional): How many episodes to sample for 
            each theta configuration. Defaults to 10.
            
            env (BaseEnv, optional): The environment in which the agent has to 
            act. Defaults to None.
            
            policy (BasePolicy, optional): The parametric policy to use. 
            Defaults to None.
            
            data_processor (IdentityDataProcessor, optional): the object in 
            charge of transforming the state into a feature vector. Defaults to 
            None.
            
            directory (str, optional): where to save the results
        """
        # Arguments
        self.lr = lr

        assert initial_rho is not None, "[ERROR] No initial hyperpolicy."
        self.rho = np.array(initial_rho, dtype=float)

        self.ite = ite
        self.batch_size = batch_size
        self.episodes_per_theta = episodes_per_theta

        assert env is not None, "[ERROR] No env provided."
        self.env = env

        assert policy is not None, "[ERROR] No policy provided."
        self.policy = policy

        assert data_processor is not None, "[ERROR] No data processor."
        self.data_processor = data_processor

        self.directory = directory
        self.verbose = verbose

        # Other paraemeters
        dim = len(self.rho[RhoElem.MEAN])
        self.thetas = np.zeros((self.batch_size, dim))
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=float)
        self.performance_idx_theta = np.zeros((ite, batch_size), dtype=float)
        self.epsilon = 0

        # Best saving parameters
        self.best_theta = np.zeros(dim, dtype=float)
        self.best_rho = self.rho
        self.best_performance = -np.inf
        return

    def learn(self) -> None:
        """Learning function"""
        for i in tqdm(range(self.ite)):
            for j in range(self.batch_size):
                # Sample theta
                self.sample_theta(index=j)

                # Collect Trajectories
                sample_mean = np.zeros(self.episodes_per_theta)
                for z in range(self.episodes_per_theta):
                    sample_mean[z] = self.collect_trajectory(
                        params=self.thetas[j, :])
                perf = np.mean(sample_mean)
                self.performance_idx_theta[i, j] = perf

                # Try to update the best config
                self.update_best(current_perf=perf, params=self.thetas[j, :])

            # Update parameters
            self.performance_idx[i] = np.mean(self.performance_idx_theta[i, :])
            self.update_rho()

            # Update time counter
            self.time += 1
            if self.verbose:
                print(f"rho perf: {self.performance_idx}")
                print(f"theta perf: {self.performance_idx_theta}")
        return

    def update_rho(self) -> None:  # FIXME
        """This function modifies the self.rho vector, by updating via the 
        estimated gradient."""
        # Take the performance of the whole batch
        batch_perf = self.performance_idx_theta[self.time, :]

        # Loop over the rho elements
        for id in range(len(self.rho[RhoElem.MEAN])):
            cur_mean_vec = self.rho[RhoElem.MEAN, id] * np.ones(self.batch_size)
            cur_std_vec = self.rho[RhoElem.STD, id] * np.ones(self.batch_size)
            cur_real_std_vec = np.exp(cur_std_vec)

            log_nu_rho_mean = (self.thetas[:, id] - cur_mean_vec) / (cur_real_std_vec**2)
            log_nu_rho_std = (((self.thetas[:, id] - cur_mean_vec) ** 2) - (cur_real_std_vec ** 2)) / (cur_real_std_vec ** 3)

            grad_m = (log_nu_rho_mean * batch_perf)
            grad_s = (log_nu_rho_std * batch_perf)

            self.rho[RhoElem.MEAN, id] += self.lr * np.mean(grad_m)
            self.rho[RhoElem.STD, id] += self.lr * np.mean(grad_s)

            if self.verbose:
                print(f"MEANs: {cur_mean_vec[0]} - STD: {cur_std_vec[0]}")
                print(f"LOG MEANs: {log_nu_rho_mean}")
                print(f"LOG STDs: {log_nu_rho_std}")
                print(f"GRAD MEANs: {np.mean(grad_m)} - GRAD STDs: {np.mean(grad_s)}")
                print(f"RHO: mean => {self.rho[RhoElem.MEAN, id]} - std => {np.exp(self.rho[RhoElem.STD, id])}")
                # input("Press ENTER to CONTINUE...")
        return

    def sample_theta(self, index: int) -> None:
        """
        Summary:
            This function modifies the self.thetas vector, by sampling parameters
            from the current rho configuration. Rho is assumed to be of the form: 
            "[[means...], [log(std_devs)...]]".
        Args:
            index (int): the current index of the thetas vector
        """
        for id in range(len(self.rho[RhoElem.MEAN])):
            self.thetas[index, id] = np.random.normal(
                loc=self.rho[RhoElem.MEAN, id], scale=np.exp(self.rho[RhoElem.STD, id])
            )
        return

    def collect_trajectory(self, params: np.array) -> float:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta 
            configuration.
        Args:
            params (np.array): the current sampling of theta values
        Returns:
            float: the discounted reward of the trajectory
        """
        # reset the environment
        self.env.reset()

        # initialize parameters
        perf = 0
        self.policy.set_parameters(thetas=params)

        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state

            # transform the state
            features = self.data_processor.transform(state=state)

            # select the action
            a = self.policy.draw_action(state=features)

            # play the action
            _, rew, _ = self.env.step(action=a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew

            if self.verbose:
                print("******************************************************")
                print(f"ACTION: {a.radius} - {a.theta}")
                print(f"FEATURES: {features}")
                print(f"REWARD: {rew}")
                print(f"PERFORMANCE: {perf}")
                print("******************************************************")

        return perf

    def update_best(self, current_perf: float, params: np.array) -> None:
        """
        Summary:
            Function updating the best configuration found so far
        Args:
            current_perf (float): current performance value to evaluate
            params (np.array): the current sampling of theta values
        """
        if current_perf > self.best_performance:
            self.best_rho = self.rho
            self.best_theta = params
            self.best_performance = current_perf
            print("-----------------------------------------------------------")
            print(f"New best RHO: {self.best_rho}")
            print(f"New best THETA: {self.best_theta}")
            print(f"New best PERFORMANCE: {self.best_performance}")
            print("-----------------------------------------------------------")
        return

    def save_results(self) -> None:
        """Function saving the results of the training procedure"""
        # Create the dictionary with the useful info
        results = {
            "performance_rho": self.performance_idx.tolist(),
            "performance_thetas_per_rho": self.performance_idx_theta.tolist(),
            "best_theta": self.best_theta.tolist(),
            "best_rho": self.best_rho.tolist()
        }

        # Save the json
        name = self.directory + "/pgpe_results.json"
        if not os.path.exists(os.path.dirname(name)):
            try:
                os.makedirs(os.path.dirname(name))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
