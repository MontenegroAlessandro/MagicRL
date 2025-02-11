"""Utils for the Algorithms"""
import os
import errno

import numpy as np
import torch
from joblib import Parallel, delayed


class RhoElem:
    MEAN = 0
    STD = 1


class LearnRates:
    PARAM = 0
    LAMBDA = 1
    ETA = 2


class TrajectoryResults:
    PERF = 0
    RewList = 1
    ScoreList = 2
    CostInfo = 3

class OffPolicyTrajectoryResults:
    PERF = 0
    RewList = 1
    ScoreList = 2
    StateList = 3
    ActList = 4

class ParamSamplerResults:
    THETA = 0
    PERF = 1
    COST = 2


def check_directory_and_create(dir_name: str = None) -> None:
    """
    Summary:
        This function checks if a directory exists.
        If it is not the case, it creates the directory.
    Args:
          dir_name (str): the name of the directory to check.
          Default is None.
    """
    if not os.path.exists(os.path.dirname(dir_name + "/")):
        try:
            os.makedirs(os.path.dirname(dir_name + "/"))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return


def is_numpy(arr) -> bool:
    return isinstance(arr, np.ndarray)


def is_tensor(arr) -> bool:
    return isinstance(arr, torch.Tensor)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array."""
    if is_numpy(tensor):
        return tensor
    return tensor.detach().numpy()


def numpy_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    if is_tensor(arr):
        return arr
    return torch.tensor(arr)

#matrix shift: currently works with negative shifts
def matrix_shift(arr, num, fill_value=np.nan):
    """Helper function to shift array elements vertically
    Positive num shifts down, negative shifts up"""
    result = np.empty_like(arr)

    result[:num] = arr[-num:]
    result[num:] = fill_value

    return result

def compute_trajectory_log_sum(policy, state_sequence, action_sequence):
    """Helper function to compute log sum for a single trajectory"""
    return np.sum([policy.compute_log_pi(np.array(s), np.array(a)) 
                   for s, a in zip(state_sequence, action_sequence)], dtype=np.float64)
