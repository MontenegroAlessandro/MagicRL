"""
Summary: Test for PG on continuous cart pole environment
Author: @MontenegroAlessandro
Date 7/12/2023
todo -> create configuration files
"""
# Libraries
from envs import *
from policies import LinearGaussianPolicy
from algorithms import PolicyGradient
from data_processors import IdentityDataProcessor
from art import *
import envs.utils
import copy

"""Global Vars"""
# general
dir = "/Users/ale/results/pg/pg_test"

# environment
horizon = 100
gamma = 1

# algorithm
DEBUG = False
NATURAL = False
LR_STRATEGY = "adam"
PARALLEL_FLAG = True

# test
test_ite = 10

"""Environment"""
env = ContCartPole(horizon=horizon, gamma=gamma)

"""Data Processor"""
dp = IdentityDataProcessor()

"""Policy"""
pol = LinearGaussianPolicy(
    parameters=[1] * 4,
    std_dev=0.3,
    action_bounds=[-10, 10]
)

"""Algorithm"""
alg_parameters = dict(
    lr=[1e-1] * 4,
    lr_strategy=LR_STRATEGY,
    estimator_type="REINFORCE",
    initial_theta=[1] * 4,
    ite=1000,
    batch_size=5,
    env=env,
    policy=pol,
    data_processor=dp,
    directory=dir,
    verbose=DEBUG,
    natural=NATURAL,
    checkpoint_freq=100,
    parallel_computation=PARALLEL_FLAG
)

alg = PolicyGradient(**alg_parameters)


if __name__ == "__main__":
    # Learn phase
    print(text2art("== PG v1.0 =="))
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)
