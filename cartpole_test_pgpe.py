# Libraries
from envs import *
from policies import LinearPolicy
from algorithms import PGPE
from data_processors import IdentityDataProcessor
from art import *

"""Global Vars"""
# general
dir = "/Users/ale/results/pgpe/pg_test_cartpole"

# environment
horizon = 100
gamma = 1

# algorithm
DEBUG = False
NATURAL = False
LR_STRATEGY = "adam"
LEARN_STD = False
ITE = 1000

# test
test_ite = 10

"""Environment"""
env = ContCartPole(horizon=horizon, gamma=gamma)

"""Data Processor"""
dp = IdentityDataProcessor()

"""Policy"""
pol = LinearPolicy(
    parameters=[1] * 4,
    action_bounds=[-10, 10]
)

"""Algorithms"""
hp = np.zeros((2, 4))
hp[0] = [0.5] * 4
hp[1] = [0.001] * 4
alg_parameters = dict(
    lr=[1e-1],
    initial_rho=hp,
    ite=ITE,
    batch_size=20,
    episodes_per_theta=1,
    env=env,
    policy=pol,
    data_processor=dp,
    directory=dir,
    verbose=DEBUG,
    natural=NATURAL,
    checkpoint_freq=100,
    lr_strategy=LR_STRATEGY,
    learn_std=LEARN_STD,
    std_decay=0,
    std_min=1e-4,
    n_jobs_param=6,
    n_jobs_traj=1
)
alg = PGPE(**alg_parameters)

if __name__ == "__main__":
    # Learn phase
    print(text2art("== PGPE v2.0 =="))
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)
