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
horizon = 50
gamma = 1
grid_size = 10
num_basis = 3
dim_state = 2
dir = "~/PyProjects/results/cpgpe_exp/test"
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

# Environment
env = GridWorldEnvCont(
    horizon=horizon, 
    gamma=gamma, 
    grid_size=grid_size, 
    reward_type="linear",
    render=RENDER,
    dir=dir,
    # obstacles=[square],
    # init_state=[1, 12]
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
hp = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
               [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001]])
alg = PGPE(
    lr=1e-3,
    initial_rho=hp,
    ite=100,
    batch_size=20,
    episodes_per_theta=20,
    env=env,
    policy=pol,
    data_processor=dp,
    directory=dir,
    verbose=DEBUG
)

if __name__ == "__main__":
    # Learn phase
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)

    # Test phase
    pol.set_parameters(thetas=alg.best_theta)
    env.reset()
    perf = 0
    env.dir = dir
    env.render = True
    perfs = []
    for i in range(10):
        env.reset()
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
    
    print(f"Evaluation Performance: {np.mean(perfs)} +/- {np.std(perfs)}")
    env.reset()
    