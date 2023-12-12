"""Test for CPGPE on continuous grid world environment"""
# todo -> create configuration files

# Libraries
from envs import *
from policies import GWPolicy
from algorithms import CvarPGPE, CPGPE
from data_processors import GWDataProcessorRBF
from algorithms.utils import RhoElem
from art import *
import envs.utils
import copy

"""Global Vars"""
# general
dir = "../../results/cost_pgpe_exp/cpgpe_test_3"

# environment
horizon = 150
gamma = 1
grid_size = 10

# state representation
num_basis = 8  # fixme
dim_state = 2

# algorithm
RENDER = False
DEBUG = False
NATURAL = False
LEARN_STD = False
LR_STRATEGY = "adam"
BASELINE = False

# test
test_ite = 10

"""Design a U near the goal"""
U_obstacle = envs.utils.design_u_obstacle(grid_size=grid_size)

"""Environment"""
sampling_args = {
    "n_samples": 1,
    "density": 3,
    "radius": 2,
    "noise": 0.1,
    "left_lim": 0,
    "right_lim": np.pi
}
env = GridWorldEnvCont(
    horizon=horizon,
    gamma=gamma,
    grid_size=grid_size,
    reward_type="linear",
    render=RENDER,
    dir=None,
    obstacles=U_obstacle,
    pacman=False,
    goal_tol=0.3,
    obstacles_strict_flag=False,
    use_costs=True,
    init_state=None,
    sampling_strategy="sphere",
    sampling_args=copy.deepcopy(sampling_args)
)
# init_state [3, 5] is starting at the left of the U obstacle

"""Data Processor"""
dp = GWDataProcessorRBF(
    num_basis=num_basis,
    grid_size=grid_size,
    std_dev=0.8
)

"""Policy"""
pol = GWPolicy(
    thetas=[1] * dim_state * num_basis,
    dim_state=num_basis * dim_state
)

"""Algorithm"""
# open an old configuration
# file_name = path = "/Users/ale/results/cpgpe_exp/cpgpe_test_am_13_cont2/cpgpe_results.json"
# file = open(file_name)
# data = json.load(file)

# FIXME dim_state * num_basis * 2
hp = np.zeros((2, dim_state * num_basis))
# hp[0] = [0.5] * dim_state * num_basis
# hp[0] = np.array(data["final_rho"][RhoElem.MEAN])
# hp[1] = [0.001] * dim_state * num_basis
# hp[1] = [0.1] * dim_state * num_basis

cpgpe_parameters = dict(
    lr=[1e-1, 1e-1], # rho, lambda, eta
    initial_rho=hp,
    ite=1000,
    batch_size=30,
    episodes_per_theta=3,
    env=env,
    policy=pol,
    data_processor=dp,
    directory=dir,
    verbose=DEBUG,
    natural=NATURAL,
    constraints=[.1, .1],
    cost_mask=[True, False],
    learn_std=LEARN_STD,
    init_lambda=np.array([5, 5]), #7 wow
    lr_strategy=LR_STRATEGY,
    resume_from=None,
    checkpoint_freq=100
)

cvarpgpe_parameters = copy.deepcopy(cpgpe_parameters)
cvarpgpe_parameters["init_eta"] = np.array([0, 0])
cvarpgpe_parameters["conf_values"] = np.array([0.95, 0.95])

alg = CPGPE(**cpgpe_parameters)
# alg = CvarPGPE(**cvarpgpe_parameters)


if __name__ == "__main__":
    # Learn phase
    print(text2art("== CPGPE v1.0 =="))
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)

    # Test phase
    # pol.set_parameters(thetas=alg.best_theta)
    # pol.set_parameters(thetas=alg.sample_theta_from_best())
    # pol.set_parameters(thetas=alg.best_rho[RhoElem.MEAN])
    pol.set_parameters(thetas=alg.rho[RhoElem.MEAN])
    env.reset()
    perf = 0
    perfs = []

    # Set the saving image logic
    env.dir = dir
    env.render = True

    # test start
    print(text2art("Test Start"))
    for i in range(test_ite):
        for t in range(env.horizon):
            # retrieve the state
            state = env.state

            # transform the state
            features = dp.transform(state=state)

            # select the action
            a = pol.draw_action(state=features)

            # play the action
            _, rew, _, _ = env.step(action=a)

            # update the performance index
            perf += (env.gamma ** t) * rew

        perfs.append(perf)
        perf = 0
        env.reset()
    print(f"BEST THETA: {alg.best_theta}")
    print(f"BEST RHO: {alg.best_rho}")
    print(f"Evaluation Performance: {np.mean(perfs)} +/- {np.std(perfs)}")
    env.reset()
