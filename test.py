"""
Summary: Test for PGPE on continuous grid world environment
Author: @MontenegroAlessandro
Date 20/7/2023
"""
# Libraries
from envs import *
from policies import GWPolicy
from algorithms import PGPE
from data_processors import GWDataProcessorRBF

# Global Vars
horizon = 30
gamma = 1
grid_size = 20
num_basis = 3
dim_state = 2

# Obstacles
square = Obstacle(
    type="square",
    features={"p1": Position(grid_size/2-1, grid_size/2+1),
              "p2": Position(grid_size/2+1, grid_size/2+1),
              "p3": Position(grid_size/2+1, grid_size),
              "p4": Position(grid_size/2-1, grid_size)}
)

# Environment
env = GridWorldEnvCont(
    horizon=horizon, 
    gamma=gamma, 
    grid_size=grid_size, 
    reward_type="sparse", 
    render=True,
    dir="../../Desktop/cpgpe_exp/test",
    obstacles=[square],
    #init_state=[1, 12]
)

# Data Processor
dp = GWDataProcessorRBF(
    num_basis=num_basis,
    grid_size=grid_size,
    std_dev=0.5
)

# Policy
pol = GWPolicy(
    thetas=[1]*dim_state*num_basis,
    dim_state=num_basis*dim_state
)

# Algorithm
hp = np.array([
    [1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1], [5, 0.1], [6, 0.1],
    [1, 0.1], [2, 0.1], [3, 0.1], [4, 0.1], [5, 0.1], [6, 0.1]
])
alg = PGPE(
    lr=1e-3,
    initial_rho=hp,
    ite=100,
    batch_size=10,
    episodes_per_theta=10,
    env=env,
    policy=pol,
    data_processor=dp,
)

if __name__ == "__main__":
    alg.learn()
    print(alg.performance_idx)
    