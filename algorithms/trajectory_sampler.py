"""
Summary: Trajectory Sampler Implementation
Author: @MontenegroAlessandro
Date: 6/12/2023
"""
# Libraries
from envs import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor
from utils import RhoElem
import numpy as np
import copy


def pg_sampling_worker(
        env=None,
        pol=None,
        dp=None,
        params: np.array = None,
        starting_state=None
) -> list:
    trajectory_sampler = TrajectorySampler(env=env, pol=pol, data_processor=dp)
    res = trajectory_sampler.collect_trajectory(params=params, starting_state=starting_state)
    return res


class ParameterSampler:
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            data_processor: BaseProcessor = None,
            episodes_per_theta: int = 1,
            n_jobs: int = 1
    ) -> None:
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

    def collect_trajectories(self, params: np.array):
        # sample a parameter configuration
        dim = len(params[RhoElem.MEAN])
        thetas = np.zeros(dim, dtype=np.float128)
        for i in range(dim):
            thetas[i] = np.random.normal(
                thetas[RhoElem.MEAN, i],
                np.float128(np.exp(thetas[RhoElem.STD]))
            )

        # collect performances over the sampled parameter configuration
        res = 0
        return


class TrajectorySampler:
    def __init__(
            self, env: BaseEnv = None,
            pol: BasePolicy = None,
            data_processor: BaseProcessor = None
    ) -> None:
        err_msg = "[PGTrajectorySampler] no environment provided!"
        assert env is not None, err_msg
        self.env = env

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

        # initialize parameters
        perf = 0
        rewards = np.zeros(self.env.horizon, dtype=np.float128)
        scores = np.zeros((self.env.horizon, self.pol.dim), dtype=np.float128)
        if params is not None:
            self.pol.set_parameters(thetas=params)

        # act
        for t in range(self.env.horizon):
            # retrieve the state
            state = self.env.state

            # transform the state
            features = self.dp.transform(state=state)

            # select the action
            a = self.pol.draw_action(state=features)
            score = self.pol.compute_score(state=features, action=a)

            # play the action
            _, rew, abs, _ = self.env.step(action=a)

            # update the performance index
            perf += (self.env.gamma ** t) * rew

            # update the vectors of rewards and scores
            rewards[t] = rew
            scores[t, :] = score

            if abs:
                if t < self.env.horizon - 1:
                    rewards[t+1:] = 0
                    scores[t+1:] = 0
                break

        return [perf, rewards, scores]
