"""
Summary: Test for PGPE on continuous grid world environment
Author: @MontenegroAlessandro
Date 20/7/2023
"""
import envs.utils
# Libraries
from envs import *
from policies import GWPolicy
from algorithms import PGPE
from data_processors import GWDataProcessorRBF
from algorithms.utils import RhoElem
from art import *

# Global Vars
horizon = 150
gamma = 1
grid_size = 10
num_basis = 4
dim_state = 2
dir = "../../../results/cpgpe_exp/pgpe_test_obs"
RENDER = False
DEBUG = False

# Obstacles
square = Obstacle(
    type="square",
    features={"p1": Position(grid_size/2-1, grid_size/2+1),
              "p2": Position(grid_size/2+1, grid_size/2+1),
              "p3": Position(grid_size/2+1, grid_size),
              "p4": Position(grid_size/2-1, grid_size)}
)

# Design a U near the goal
U_obstacle = envs.utils.design_u_obstacle(grid_size=grid_size)

# Environment
env = GridWorldEnvCont(
    horizon=horizon, 
    gamma=gamma, 
    grid_size=grid_size, 
    reward_type="linear",
    render=RENDER,
    dir=None,
    obstacles=U_obstacle,
    # init_state=[0, 0],
    pacman=False,
    goal_tol=0.5
)

# Data Processor
dp = GWDataProcessorRBF(
    num_basis=num_basis,
    grid_size=grid_size,
    std_dev=0.8
)

# Policy
pol = GWPolicy(
    thetas=[1]*dim_state*num_basis,
    dim_state=num_basis*dim_state
)

# Algorithm
# FIXME dim_state * num_basis * 2
hp = np.zeros((2, dim_state * num_basis))
hp[0] = [0.5] * dim_state * num_basis
hp[1] = [0.001] * dim_state * num_basis

alg = PGPE(
    lr=[1e-3],
    initial_rho=hp,
    ite=100,
    batch_size=50,
    episodes_per_theta=100,
    env=env,
    policy=pol,
    data_processor=dp,
    directory=dir,
    verbose=DEBUG,
    natural=True
)

if __name__ == "__main__":
    # Learn phase
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)

    # Test phase
    pol.set_parameters(thetas=alg.best_theta)
    # pol.set_parameters(thetas=alg.sample_theta_from_best())
    # pol.set_parameters(thetas=alg.best_rho[RhoElem.MEAN])
    env.reset()
    perf = 0
    perfs = []
    
    # Set the saving image logic
    env.dir = dir
    env.render = True
    
    # test start
    print(text2art("Test Start"))
    for i in range(10):
        for t in range(env.horizon):
            # retrieve the state
            state = env.state

            # transform the state
            features = dp.transform(state=state)

            # select the action
            a = pol.draw_action(state=features)

            # play the action
            _, rew, _ = env.step(action=a)

            # update the performance index
            perf += (env.gamma ** t) * rew
        
        perfs.append(perf)
        perf = 0
        env.reset()
    print(f"BEST THETA: {alg.best_theta}")
    print(f"BEST RHO: {alg.best_rho}")
    print(f"Evaluation Performance: {np.mean(perfs)} +/- {np.std(perfs)}")
    env.reset()
    