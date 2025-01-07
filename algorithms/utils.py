"""Utils for the Algorithms"""
import os
import errno

import numpy as np
import torch


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

class TrajectoryResults2:
    PERF = 0
    RewList = 1
    ScoreList = 2
    Info = 3


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
