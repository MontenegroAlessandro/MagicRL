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
# from mushroom_rl import COPDAC_Q


# Algorithm implementation
class COPDAC_Q:
    """DPG implememntation."""
    def __init__(self) -> None:
        pass