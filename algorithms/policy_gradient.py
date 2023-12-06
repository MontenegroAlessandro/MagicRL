"""
Summary: Policy Gradient Implementation
Author: @MontenegroAlessandro
Date: 6/12/2023
"""
# Libraries
import numpy as np
from envs.base_env import BaseEnv
from policies import BasePolicy
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import RhoElem, LearnRates, check_directory_and_create
import json, io, os, errno
from tqdm import tqdm
import copy
from adam.adam import Adam


# Class Implementation
class PolicyGradient:
    """This Class implements Policy Gradient Algorithms via REINFORCE or GPOMDP."""
    def __init__(
            self, lr: float = 1e-3,
            lr_strategy: str = "constant",
            estimator_type: str = "REINFORCE",
            initial_theta: np.array = None,
            ite: int = 100,
            batch_size: int = 1,
            env: BaseEnv = None,
            policy: BasePolicy = None,
            data_processor: BaseProcessor = IdentityDataProcessor(),
            directory: str = "",
            verbose: bool = False,
            natural: bool = False,
            checkpoint_freq: int = 1
    ) -> None:
        # Class' parameter with checks
        err_msg = "[PG] lr value cannot be negative!"
        assert lr > 0, err_msg
        self.lr = lr

        err_msg = "[PG] lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy

        err_msg = "[PG] estimator_type not valid!"
        assert estimator_type in ["REINFORCE", "GPOMDP"], err_msg
        self.estimator_type = estimator_type

        err_msg = "[PG] initial_theta has not been specified!"
        assert initial_theta is not None, err_msg
        self.thetas = np.array(initial_theta)
        self.dim = len(self.thetas)

        assert env is not None
        self.env = env

        assert policy is not None
        self.policy = policy

        assert data_processor is not None
        self.data_processor = data_processor

        check_directory_and_create(dir_name=directory)
        self.directory = directory

        # Other class' parameters
        self.ite = ite
        self.batch_size = batch_size
        self.verbose = verbose
        self.natural = natural
        self.checkpoint_freq = checkpoint_freq

        # Useful structures
        self.theta_history = np.zeros((self.ite, self.dim), dtype=float)
        self.time = 0
        self.performance_idx = np.zeros(ite, dtype=float)
        self.best_theta = np.zeros(self.dim, dtype=float)
        self.best_performance_theta = -np.inf
        return
