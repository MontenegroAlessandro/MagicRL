"""Trajectory Sampler Implementation"""
from sympy.strategies.branch.traverse import top_down

# Libraries
from envs import BaseEnv
from policies.JAX_policies.base_policy_jax import BasePolicyJAX
from data_processors import BaseProcessor
from algorithms.utils import RhoElem, TrajectoryResults
from joblib import Parallel, delayed
import numpy as np
import copy
from algorithms.utils import PolicyGradientAlgorithms
from jax import jit


def pg_sampling_worker(
        env: BaseEnv = None,
        pol: BasePolicyJAX = None,
        dp: BaseProcessor = None,
        params: np.array = None,
        starting_state: np.array = None,
        alg : PolicyGradientAlgorithms = None
) -> list:
    """
    Worker collecting a single trajectory.

    Args:
        env (BaseEnv, optional): the env to employ. Defaults to None.
        
        pol (BasePolicyJAX, optional): the policy to play. Defaults to None.
        
        dp (Baseprocessor, optional): the data processor to employ. 
        Defaults to None.
        
        params (np.array, optional): the parameters to plug into the policy. 
        Defaults to None.
        
        starting_state (np.array, optional): the state to which the env should 
        be initialized. Defaults to None.

    Returns:
        list: [performance, reward, scores]
    """
    # pol = copy.deepcopy(pol)
    trajectory_sampler = TrajectorySampler(env=env, pol=pol, data_processor=dp, alg=alg)
    res = trajectory_sampler.collect_trajectory(params=params, starting_state=starting_state)
    return res


def pgpe_sampling_worker(
        env: BaseEnv = None,
        pol: BasePolicyJAX = None,
        dp: BaseProcessor = None,
        params: np.array = None,
        episodes_per_theta: int = None,
        n_jobs: int = None,
        alg: PolicyGradientAlgorithms = None
) -> np.array:
    """Worker collecting trajectories for muliple sampling of parameters from the hyperpolicy.

    Args:
        env (BaseEnv, optional): the env to use. Defaults to None.
        
        pol (BasePolicyJAX, optional): the policy to play. Defaults to None.
        
        dp (BaseProcessor, optional): the data processor to use. 
        Defaults to None.
        
        params (np.array, optional): the parameter of the hyper.policy. 
        Defaults to None.
        
        episodes_per_theta (int, optional): how many episodes to evaluate for 
        each sampled parameter. Defaults to None.
        
        n_jobs (int, optional): how many parallel trajectories to evaluate 
        in parallel. Defaults to None.

    Returns:
        np.array: [parameters, performance]
    """
    parameter_sampler = ParameterSampler(
        env=env,
        pol=pol,
        data_processor=dp,
        episodes_per_theta=episodes_per_theta,
        n_jobs=n_jobs,
        alg = alg
    )
    res = parameter_sampler.collect_trajectories(params=params)
    return res


class ParameterSampler:
    """Sampler for PGPE."""
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicyJAX = None,
            data_processor: BaseProcessor = None,
            episodes_per_theta: int = 1,
            n_jobs: int = 1,
            alg: PolicyGradientAlgorithms = None
    ) -> None:
        """
        Summary:
            Initialization.

        Args:
            env (BaseEnv, optional): the env to employ. Defaults to None.
            
            pol (BasePolicyJAX, optional): the poliy to play. Defaults to None.
            
            data_processor (BaseProcessor, optional): the data processor to use. 
            Defaults to None.
            
            episodes_per_theta (int, optional): how many trajectories to 
            evalluate for each sampled theta. Defaults to 1.
            
            n_jobs (int, optional): how many theta sample (and evaluate) 
            in parallel. Defaults to 1.
        """
        err_msg = "[PGPETrajectorySampler] no environment provided!"
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PGPETrajectorySampler] no policy provided!"
        assert pol is not None, err_msg
        self.pol = pol

        err_msg = "[PGPETrajectorySampler] no data_processor provided!"
        assert data_processor is not None, err_msg
        self.dp = data_processor

        self.episodes_per_theta = episodes_per_theta
        self.trajectory_sampler = TrajectorySampler(
            env=self.env,
            pol=self.pol,
            data_processor=self.dp,
            alg=alg
        )
        self.n_jobs = n_jobs

        self.alg = alg

        return

    def collect_trajectories(self, params: np.array) -> list:
        """
        Summary:
            Collect the trajectories for a sampled parameter configurations.

        Args:
            params (np.array): hyper-policy configuration.

        Returns:
            list: [params, performance]
        """
        # sample a parameter configuration
        dim = len(params[RhoElem.MEAN])
        thetas = np.zeros(dim, dtype=np.float64)
        for i in range(dim):
            thetas[i] = np.random.normal(
                params[RhoElem.MEAN, i],
                np.float64(np.exp(params[RhoElem.STD, i]))
            )

        # collect performances over the sampled parameter configuration
        if self.n_jobs == 1:
            raw_res = []
            for i in range(self.episodes_per_theta):
                raw_res.append(self.trajectory_sampler.collect_trajectory(
                    params=thetas, starting_state=None)
                )
        else:
            worker_dict = dict(
                env=copy.deepcopy(self.env),
                pol=copy.deepcopy(self.pol),
                dp=copy.deepcopy(self.dp),
                params=copy.deepcopy(thetas),
                starting_state=None
            )
            # build the parallel functions
            delayed_functions = delayed(pg_sampling_worker)

            # parallel computation
            raw_res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**worker_dict) for _ in range(self.episodes_per_theta)
            )

        # keep just the performance over each trajectory
        res = np.zeros(self.episodes_per_theta, dtype=np.float64)
        for i, elem in enumerate(raw_res):
            res[i] = elem[TrajectoryResults.PERF]

        return [thetas, res]


class TrajectorySampler:
    """Trajectory sampler for PolicyGradient methods."""
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicyJAX = None,
            fun = None,
            data_processor: BaseProcessor = None,
            alg: PolicyGradientAlgorithms = None
    ) -> None:
        """
        Summary:
            Initialization.

        Args:
            env (BaseEnv, optional): the env to use. Defaults to None.
            
            pol (BasePolicyJAX, optional): the policy to play. Defaults to None.
            
            data_processor (BaseProcessor, optional): the data processor to use. 
            Defaults to None.
        """
        err_msg = "[PGTrajectorySampler] no environment provided!"
        assert env is not None, err_msg
        self.env = env

        self.alg = alg

        err_msg = "[PGTrajectorySampler] no policy provided!"
        assert pol is not None, err_msg
        self.pol = pol

        err_msg = "[PGTrajectorySampler] no data_processor provided!"
        assert data_processor is not None, err_msg
        self.dp = data_processor

        return

    def collect_trajectory(
            self, params: np.array = None, starting_state=None
    ) -> list:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta
            configuration.
        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): teh starting state for the iterations
        Returns:
            list of:
                float: the discounted reward of the trajectory
                np.array: vector of all the rewards
                np.array: vector of all the scores
        """
        # reset the environment
        self.env.reset()
        if starting_state is not None:
            self.env.state = copy.deepcopy(starting_state)

        if params is not None:
            self.pol.set_parameters(thetas=params)

        # initialize parameters
        np.random.seed()
        perf = 0
        rewards = np.zeros(self.env.horizon, dtype=np.float64)
        scores = np.zeros((self.env.horizon, self.pol.tot_params), dtype=np.float64)

        if params is not None:
            self.pol.set_parameters(thetas=params)

        # collect states and action for the whole trajectory
        actions = []
        states = []

        for t in range(self.env.horizon):
            state = self.env.state
            features = self.dp.transform(state=state)
            action = self.pol.draw_action(state=features)

            if self.alg != PolicyGradientAlgorithms.PGPE:
                states.append(features)
                actions.append(action)

            _, reward, done, _ = self.env.step(action=action)

            perf += (self.env.gamma ** t) * reward
            rewards[t] = reward

            if done:
                rewards[t + 1:] = 0
                break

        states = np.array(states, dtype=np.float64)
        actions = np.array(actions, dtype=np.float64)


        if self.alg != PolicyGradientAlgorithms.PGPE:
            # compute the scores for the whole trajectory
            scores = self.pol.compute_score(state=states, action=actions)
            # todo:  may be necessary to add a zero in the last position of the scores


        return [perf, rewards, scores]
