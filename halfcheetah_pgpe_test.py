# Libraries
from envs import *
from policies import LinearPolicy
from algorithms import PGPE
from data_processors import IdentityDataProcessor
from art import *

"""Global Vars"""
# general
dir = "/Users/ale/results/pgpe/pgpe_test_halfcheetah"
MODE = "learn_test"

# environment
horizon = 100
gamma = 1
RENDER = False

# algorithm
DEBUG = False
NATURAL = False
LR_STRATEGY = "adam"
LEARN_STD = False
ITE = 2000
BATCH = 30
EPISODES = 1
N_JOBS_PARAM = 6
N_JOBS_TRAJ = 1
STD_DECAY = 0   # 1e-5
STD_MIN = 1e-4

# test
test_ite = 200
num_test = 10

"""Environment"""
env = HalfCheetah(horizon=horizon, gamma=gamma, render=RENDER)
s_dim = env.state_dim
a_dim = env.action_dim

"""Data Processor"""
dp = IdentityDataProcessor()

"""Policy"""
pol = LinearPolicy(
    parameters=np.ones((a_dim, s_dim)),
    action_bounds=env.action_bounds,
    multi_linear=True
)

"""Algorithms"""
hp = np.zeros((2, s_dim * a_dim))
hp[0] = [0.5] * (s_dim * a_dim)
hp[1] = [0.001] * (s_dim * a_dim)
alg_parameters = dict(
    lr=[1e-1],
    initial_rho=hp,
    ite=ITE,
    batch_size=BATCH,
    episodes_per_theta=EPISODES,
    env=env,
    policy=pol,
    data_processor=dp,
    directory=dir,
    verbose=DEBUG,
    natural=NATURAL,
    checkpoint_freq=100,
    lr_strategy=LR_STRATEGY,
    learn_std=LEARN_STD,
    std_decay=STD_DECAY,
    std_min=STD_MIN,
    n_jobs_param=N_JOBS_PARAM,
    n_jobs_traj=N_JOBS_TRAJ
)
alg = PGPE(**alg_parameters)

if __name__ == "__main__":
    # Learn phase
    print(text2art("== PGPE v2.0 =="))
    if MODE in ["learn", "learn_test"]:
        print(text2art("Learn Start"))
        alg.learn()
        alg.save_results()
        print(alg.performance_idx)

    # Test phase
    if MODE in ["test", "learn_test"]:
        print(text2art("** DAJE **"))
        env = HalfCheetah(horizon=horizon, gamma=gamma, render=True)
        pol.set_parameters(thetas=alg.best_theta[0])
        for _ in range(num_test):
            env.reset()
            state = env.state
            r = 0
            for i in range(test_ite):
                state, rew, _, _ = env.step(action=pol.draw_action(state))
                r += (gamma ** i) * rew
            print(f"PERFORMANCE: {r}")
