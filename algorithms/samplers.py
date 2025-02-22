"""Trajectory Sampler Implementation"""
# Libraries
from envs import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor
from algorithms.utils import RhoElem, TrajectoryResults
from joblib import Parallel, delayed
import numpy as np
import copy
import collections
import time

def pg_sampling_worker(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        dp: BaseProcessor = None,
        params: np.ndarray = None,
        starting_state: np.ndarray = None,
        starting_action: np.ndarray = None,
        pol_values: bool = False
) -> list:
    """Worker collecting a single trajectory.

    Args:
        env (BaseEnv, optional): the env to employ. Defaults to None.
        
        pol (BasePolicy, optional): the policy to play. Defaults to None.
        
        dp (Baseprocessor, optional): the data processor to employ. 
        Defaults to None.
        
        params (np.array, optional): the parameters to plug into the policy. 
        Defaults to None.
        
        starting_state (np.array, optional): the state to which the env should 
        be initialized. Defaults to None.

    Returns:
        list: [performance, reward, scores]
    """
    trajectory_sampler = TrajectorySampler(env=env, pol=pol, data_processor=dp, pol_values=pol_values)
    res = trajectory_sampler.collect_trajectory(params=params, starting_state=starting_state, starting_action=starting_action)
    return res


def off_pg_sampling_worker(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        dp: BaseProcessor = None,
        params: np.ndarray = None,
        starting_state: np.ndarray = None,
        starting_action: np.ndarray = None,
        pol_values: bool = False,
        thetas_queue: collections.deque = None,
        deterministic: bool = False
) -> list:
    """Worker collecting a single trajectory.

    Args:
        env (BaseEnv, optional): the env to employ. Defaults to None.
        
        pol (BasePolicy, optional): the policy to play. Defaults to None.
        
        dp (Baseprocessor, optional): the data processor to employ. 
        Defaults to None.
        
        params (np.array, optional): the parameters to plug into the policy. 
        Defaults to None.
        
        starting_state (np.array, optional): the state to which the env should 
        be initialized. Defaults to None.

    Returns:
        list: [performance, reward, scores]
    """
    trajectory_sampler = TrajectorySampler(env=env, pol=pol, data_processor=dp, pol_values=pol_values)
    res = trajectory_sampler.collect_off_policy_trajectory(params=params, starting_state=starting_state, starting_action=starting_action, thetas_queue=thetas_queue, deterministic=deterministic)
    return res


def pgpe_sampling_worker(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        dp: BaseProcessor = None,
        params: np.array = None,
        episodes_per_theta: int = None,
        n_jobs: int = None
) -> np.array:
    """Worker collecting trajectories for muliple sampling of parameters from the hyperpolicy.

    Args:
        env (BaseEnv, optional): the env to use. Defaults to None.
        
        pol (BasePolicy, optional): the policy to play. Defaults to None.
        
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
        n_jobs=n_jobs
    )
    res = parameter_sampler.collect_trajectories(params=params)
    return res


class ParameterSampler:
    """Sampler for PGPE."""
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            data_processor: BaseProcessor = None,
            episodes_per_theta: int = 1,
            n_jobs: int = 1
    ) -> None:
        """
        Summary:
            Initialization.

        Args:
            env (BaseEnv, optional): the env to employ. Defaults to None.
            
            pol (BasePolicy, optional): the poliy to play. Defaults to None.
            
            data_processor (BaseProcessor, optional): the data processor to use. 
            Defaults to None.
            
            episodes_per_theta (int, optional): how many trajectories to 
            evaluate for each sampled theta. Defaults to 1.
            
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
            data_processor=self.dp
        )
        self.n_jobs = n_jobs

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
        # if it is the case keep also the costs
        perf_res = np.zeros(self.episodes_per_theta, dtype=np.float64)
        cum_costs = np.zeros(
            shape=(self.episodes_per_theta, self.env.how_many_costs),
            dtype=np.float64
        )
        for i, elem in enumerate(raw_res):
            perf_res[i] = elem[TrajectoryResults.PERF]
            if self.env.with_costs:
                cum_costs[i, :] = np.array(
                    elem[TrajectoryResults.CostInfo]["cost_perf"],
                    dtype=np.float64
                )

        return [thetas, perf_res, cum_costs]


class TrajectorySampler:
    """Trajectory sampler for PolicyGradient methods."""
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            data_processor: BaseProcessor = None,
            pol_values: bool = False
    ) -> None:
        """
        Summary:
            Initialization.

        Args:
            env (BaseEnv, optional): the env to use. Defaults to None.
            
            pol (BasePolicy, optional): the policy to play. Defaults to None.
            
            data_processor (BaseProcessor, optional): the data processor to use. 
            Defaults to None.
        """
        err_msg = "[PGTrajectorySampler] no environment provided!"
        assert env is not None, err_msg
        self.env = env

        err_msg = "[PGTrajectorySampler] no policy provided!"
        assert pol is not None, err_msg
        self.pol = pol

        err_msg = "[PGTrajectorySampler] no data_processor provided!"
        assert data_processor is not None, err_msg
        self.dp = data_processor
        
        self.pol_values = pol_values

        return

    def collect_trajectory(
            self, params: np.array = None, starting_state=None, starting_action=None
    ) -> list:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta
            configuration.
        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): the starting state for the iterations
        Returns:
            list of:
                float: the discounted reward of the trajectory
                np.array: vector of all the rewards
                np.array: vector of all the scores
        """
        # reset the environment
        self.env.reset()
        if starting_state is not None:
            # self.env.state = copy.deepcopy(starting_state)
            self.env.set_state(starting_state)

        # initialize parameters
        np.random.seed()
        perf = 0
        cost_perf = np.zeros(self.env.how_many_costs)
        rewards = np.zeros(self.env.horizon, dtype=np.float64)
        costs = np.zeros(shape=(self.env.horizon, self.env.how_many_costs), dtype=np.float64)
        scores = np.zeros(shape=(self.env.horizon, self.pol.tot_params), dtype=np.float64)

        pol_values = 0

        if params is not None:
            self.pol.set_parameters(thetas=params)

        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state

            # transform the state
            features = self.dp.transform(state=state)

            # select the action
            if t == 0 and starting_action is not None:
                a = starting_action
            else:
                a = self.pol.draw_action(state=features)
            score = self.pol.compute_score(state=features, action=a)

            # play the action
            _, rew, done, info = self.env.step(action=a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew
            if self.env.with_costs:
                cost_array = copy.deepcopy(np.array(info["costs"], dtype=np.float64))
                cost_perf = cost_perf + (self.env.gamma ** t) * cost_array
                costs[t, :] = copy.deepcopy(cost_array)
                
            if self.pol_values:
                pol_values += np.log(self.pol.compute_pi(state=features, action=a))

            # update the vectors of rewards and scores
            rewards[t] = rew
            scores[t, :] = score

            if done:
                if t < self.env.horizon - 1:
                    rewards[t+1:] = 0
                    scores[t+1:, :] = 0
                    if self.env.with_costs:
                        costs[t+1:, :] = 0
                break

        # if necessary save the info of the costs
        info = None
        if self.env.with_costs:
            info = dict(
                cost_perf=copy.deepcopy(cost_perf),
                costs=copy.deepcopy(costs),
                pol_values=copy.deepcopy(pol_values),
            )
        return [perf, rewards, scores, info]
    
    def collect_off_policy_trajectory(
            self, params: np.array = None, starting_state=None, starting_action=None, 
            thetas_queue: collections.deque = None, deterministic: bool = False
    ) -> list:
        """
        Summary:
            Function collecting a trajectory reward for a particular theta
            configuration.
        Args:
            params (np.array): the current sampling of theta values
            starting_state (any): the starting state for the iterations
        Returns:
            list of:
                float: the discounted reward of the trajectory
                np.array: vector of all the rewards
                np.array: vector of all the scores
        """
        # reset the environment
        self.env.reset()
        if starting_state is not None:
            # self.env.state = copy.deepcopy(starting_state)
            self.env.set_state(starting_state)

        # initialize parameters
        np.random.seed()
        perf = 0
        rewards = np.zeros(self.env.horizon, dtype=np.float64)
        scores = np.zeros(shape=(self.env.horizon, self.pol.tot_params), dtype=np.float64)
        states = np.zeros(shape=(self.env.horizon, self.env.state_dim), dtype=np.float64)
        actions = np.zeros(shape=(self.env.horizon, self.env.action_dim), dtype=np.float64)

        
        if not deterministic:
            len_queue = len(thetas_queue)
            log_sums = np.zeros(len_queue, dtype=np.float64)
        else:
            log_sums = None

        pol_values = 0
        if params is not None:
            self.pol.set_parameters(thetas=params)

        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state
            states[t, :] = state

            # transform the state
            features = self.dp.transform(state=state)

            # select the action
            if t == 0 and starting_action is not None:
                a = starting_action
            else:
                a = self.pol.draw_action(state=features)
            actions[t, :] = a

            # play the action
            _, rew, done, info = self.env.step(action=a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew

            if self.pol_values:
                pol_values += np.log(self.pol.compute_pi(state=features, action=a))

            # update the vectors of rewards and scores
            rewards[t] = rew


            if done:
                if t < self.env.horizon - 1:
                    rewards[t+1:] = 0
                break
        
        #compute the log sum for each theta in the queue
        
        if not deterministic:
            log_sums = self.pol.compute_sum_all_log_pi(states, actions, np.array(thetas_queue))

        return [perf, rewards, scores, states, actions, log_sums]
