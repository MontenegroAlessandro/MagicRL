import json
import copy
import numpy as np
from envs import Swimmer, HalfCheetah
from policies import *

ALG = ["pgpe", "pg"][1]
ENV = ["half_cheetah", "swimmer"][0]
POL = ["linear", "nn"][1]
WHAT = ["theta", "rho"][1]
if ALG == "pgpe":
    # path = "/Users/ale/results/pgpe/pgpe_2000_swimmer_200_adam_01_nn_160_var_1"
    # path = "/Users/ale/results/pgpe/pgpe_1000_swimmer_200_adam_01_nn_clip_1344_var_1"
    # path = "/Users/ale/results/pgpe/pgpe_1000_half_cheetah_100_adam_01_linear_clip_102_var_1"
    path = "/Users/ale/results/pgpe/pgpe_2000_half_cheetah_100_adam_01_linear_clip_102_var_1"
    name = path + "/pgpe_results.json"
else:
    # path = "/Users/ale/results/pg/pg_1000_swimmer_100_adam_001_deep_gaussian_clip_160_var_1"
    # path = "/Users/ale/results/pg/pg_1000_swimmer_200_adam_001_deep_gaussian_clip_160_var_1"
    # path = "/Users/ale/results/pg/pg_1000_swimmer_200_adam_001_deep_gaussian_clip_416_var_1"
    # path = "/Users/ale/results/pg/pg_2000_half_cheetah_100_adam_001_gaussian_clip_102_var_1"
    path = "/Users/ale/results/pg/pg_2000_half_cheetah_100_adam_001_deep_gaussian_batch_100_clip_368_var_1"
    name = path + "/pg_results.json"

file = open(name)
data = json.load(file)

n_test = 10
if ENV == "half_cheetah":
    horizon = 100
    gamma = 1
    env = HalfCheetah(horizon=horizon, gamma=gamma, render=True, clip=True)
elif ENV == "swimmer":
    horizon = 200
    gamma = 1
    env = Swimmer(horizon=horizon, gamma=gamma, render=True, clip=True)
else:
    raise ValueError("boh")

if POL == "nn":
    net = nn.Sequential(
        nn.Linear(env.state_dim, 16, bias=False),
        nn.Tanh(),
        nn.Linear(16, env.action_dim, bias=False),
        nn.Tanh()
    )
    model_desc = dict(
        layers_shape=[(env.state_dim, 16), (16, env.action_dim)]
    )
    pol = NeuralNetworkPolicy(
        parameters=None,
        input_size=env.state_dim,
        output_size=env.action_dim,
        model=copy.deepcopy(net),
        model_desc=copy.deepcopy(model_desc)
    )
else:
    pol = LinearPolicy(
        parameters=np.zeros(env.action_dim * env.state_dim),
        dim_action=env.action_dim,
        dim_state=env.state_dim,
        multi_linear=True
    )

if ALG == "pgpe":
    if WHAT == "theta":
        pol.set_parameters(np.array(data["best_theta"][0]))
    else:
        pol.set_parameters(np.array(data["best_rho"][0]))
else:
    pol.set_parameters(np.array(data["best_theta"]))

perfs = []
for _ in range(n_test):
    env.reset()
    state = env.state
    r = 0
    for i in range(horizon):
        state, rew, _, _ = env.step(action=pol.draw_action(state))
        r += (gamma ** i) * rew
    print(f"PERFORMANCE: {r}")
    perfs.append(r)
print(f"{np.mean(perfs)} pm {np.std(perfs)}")
