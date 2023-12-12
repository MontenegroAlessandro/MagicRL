"""Utils for the Algorithms"""
import os
import errno


class RhoElem:
    MEAN = 0
    STD = 1


class LearnRates:
    RHO = 0
    LAMBDA = 1
    ETA = 2


class TrajectoryResults:
    PERF = 0
    RewList = 1
    ScoreList = 2


class ParamSamplerResults:
    THETA = 0
    PERF = 1


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
