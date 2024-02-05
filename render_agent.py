import json
import copy
import numpy as np
from envs import Swimmer, HalfCheetah
from policies import *

ALG = ["pgpe", "pg"][1]
ENV = ["half_cheetah", "swimmer"][1]
POL = ["linear", "nn"][0]
WHAT = ["theta", "rho"][1]

base = "/Users/ale/results/server_results/"

if ALG == "pg":
    tmp_pol = ""
    params = 0
    if POL == "linear":
        tmp_pol = "gaussian"
        if ENV == "swimmer":
            params = 16
        else:
            params = 102
    else:
        tmp_pol = "deep_gaussian"
    path = base + f"pg/pg_2000_{ENV}_100_adam_001_{tmp_pol}_batch_100_clip_{params}_var_1_trial_0"
    name = path + "/pg_results.json"
else:
    params = 0
    if POL == "linear":
        if ENV == "swimmer":
            params = 16
        else:
            params = 102
    path = base + f"pgpe/pgpe_2000_{ENV}_100_adam_01_{POL}_batch_100_clip_{params}_var_1_trial_0"
    name = path + "/pgpe_results.json"

file = open(name)
data = json.load(file)
print(name)

n_test = 10
if ENV == "half_cheetah":
    horizon = 100
    gamma = 1
    env = HalfCheetah(horizon=horizon, gamma=gamma, render=True, clip=True)
elif ENV == "swimmer":
    horizon = 100
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
