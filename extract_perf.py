import numpy as np
import json
from copy import deepcopy
from data_processors import IdentityDataProcessor
from envs import *
from policies import OldLinearPolicy
from algorithms.samplers import *
from algorithms.utils import TrajectoryResults2

# Run Parameters
exp_idx = 1
N_TRIALS = 5
LR_SIGMA_S = ["01", "01", "01"][exp_idx]
LR_SIGMA_S_A = ["01", "001", "01"][exp_idx]
LR_SIGMA_L = ["001", "001", "0001"][exp_idx] #["01", "01", "01"][exp_idx]
# LR_SIGMA_L_A = ["001", "0003", "0001"][exp_idx] #["01", "001", "001"][exp_idx]
LR_SIGMA_L_A = ["001", "001", "0001"][exp_idx] #["01", "001", "001"][exp_idx]
EXP_S = [1, 1, 1][exp_idx]
EXP_L = ["05", "05", "05"][exp_idx] #[10, 10, 10][exp_idx]
EXP_L_A = ["05", "10", "05"][exp_idx] #[10, 10, 10][exp_idx]
INNER_LOOP = [50, 200, 200][exp_idx]
HORIZON = [200, 200, 50][exp_idx]
ENV = ["ip", "swimmer", "reacher"][exp_idx]
VARS = [["1", "025", "00016", "0028"], ["1", "025", "00016", "0014"], ["025", "00676", "00004"]][exp_idx]
VARS_A = [["1", "025", "00016", "0028"], ["1", "05", "0014"], ["025", "00676", "00004"]][exp_idx]
PARAMS = [4, 16, 20][exp_idx]
PHASES = 25
INNER_LR = ["001","001","0001"][exp_idx]
INNER_LR_A = ["001", "001", "0001"][exp_idx] # ["001", "0003", "0001"][exp_idx]
SSD = 0

N_J = 5
N = 100

URGENT = 1
path = "/Users/ale/code/PyProjects/results/urgent/"
if ENV == "ip" and URGENT:
    p_path = "pes_exp_05_phases_1250_lrsigma_adam_01_pgpe_1_ip_200_adam_001_linear_batch_100_noclip_4_var_1"
    a_path = "pes_exp_05_phases_1250_lrsigma_adam_01_pg_1_ip_200_adam_001_gaussian_batch_100_noclip_4_var_1"
elif ENV == "swimmer" and URGENT:
    p_path = "pes_exp_05_phases_5000_lrsigma_adam_01_pgpe_1_swimmer_200_adam_001_linear_batch_100_noclip_16_var_1"
    # a_path = "pes_exp_05_phases_5000_lrsigma_adam_01_pg_1_swimmer_200_adam_0003_gaussian_batch_100_noclip_16_var_1"
    a_path = "pes_exp_05_phases_5000_lrsigma_adam_001_pg_1_swimmer_200_adam_001_gaussian_batch_100_noclip_16_var_1"
elif ENV == "reacher" and URGENT:
    # p_path = "pes_exp_05_phases_5000_lrsigma_adam_01_pgpe_1_reacher_50_adam_0001_linear_batch_100_noclip_20_var_1"
    p_path = "pes_exp_05_phases_5000_lrsigma_adam_0001_pgpe_1_reacher_50_adam_0001_linear_batch_100_noclip_20_var_1"
    a_path = "pes_exp_05_phases_5000_lrsigma_adam_0001_pg_1_reacher_50_adam_0001_gaussian_batch_100_noclip_20_var_1"
    # a_path = "pes_exp_05_phases_5000_lrsigma_adam_01_pg_1_reacher_200_adam_0001_gaussian_batch_100_noclip_20_var_1"

# file opening
if SSD:
    base_name = "/Volumes/FastAle/pes_pel/pes/"
else:
    base_name = f"/Users/ale/code/PyProjects/results/{ENV}/"

base_name += f"pes_exp_{EXP_S}_phases_{PHASES}_lrsigma_adam_{LR_SIGMA_S}_pgpe_{INNER_LOOP}_{ENV}_{HORIZON}_adam_{INNER_LR}_linear_batch_100_noclip_{PARAMS}_var_1/"
data = []
for i in range(N_TRIALS):
    name = base_name + f"pes_exp_{EXP_S}_phases_{PHASES}_lrsigma_adam_{LR_SIGMA_S}_pgpe_{INNER_LOOP}_{ENV}_{HORIZON}_adam_{INNER_LR}_linear_batch_100_noclip_{PARAMS}_var_1_trial_{i}/"
    with open(name + "ppes_results.json", "r") as f:
        data.append(deepcopy(json.load(f)))

if SSD:
    base_name = "/Volumes/FastAle/pes_pel/pes/"
else:
    base_name = f"/Users/ale/code/PyProjects/results/{ENV}/"
base_name += f"pel_param_exp_exp_{EXP_L}_phases_{PHASES*INNER_LOOP}_lrsigma_adam_{LR_SIGMA_L}_pgpe_1_{ENV}_{HORIZON}_adam_{INNER_LR}_linear_batch_100_noclip_{PARAMS}_var_1/"
data_pel = []
for i in range(N_TRIALS):
    name = base_name + f"pel_param_exp_exp_{EXP_L}_phases_{PHASES*INNER_LOOP}_lrsigma_adam_{LR_SIGMA_L}_pgpe_1_{ENV}_{HORIZON}_adam_{INNER_LR}_linear_batch_100_noclip_{PARAMS}_var_1_trial_{i}/"
    with open(name + "ppel_results.json", "r") as f:
        data_pel.append(deepcopy(json.load(f)))

# urgent opening
if URGENT:
    pu_data = []
    for i in range(N_TRIALS):
        name = path + p_path + "/" + p_path + f"_trial_{i}/"
        with open(name + "ppes_results.json", "r") as f:
            pu_data.append(deepcopy(json.load(f)))

# PERFORMANCE RETRIEVAL
if ENV == "swimmer":
    env = Swimmer(horizon=HORIZON, gamma=1, render=False, clip=0)
    MULTI_LINEAR = True
elif ENV == "ip":
    env = InvertedPendulum(horizon=HORIZON, gamma=1, render=False, clip=0)
    MULTI_LINEAR = False
elif ENV == "reacher":
    env = Reacher(horizon=HORIZON, gamma=1, render=False, clip=0)
    MULTI_LINEAR = True
else:
    raise ValueError(f"Invalid env name.")
s_dim = env.state_dim
a_dim = env.action_dim
dp = IdentityDataProcessor(dim_feat=env.state_dim)
tot_params = s_dim * a_dim
pol = OldLinearPolicy(
    parameters=np.zeros(tot_params),
    dim_state=s_dim,
    dim_action=a_dim,
    multi_linear=MULTI_LINEAR
)

datas = [data, data_pel, pu_data]
extracted_p_perf = np.zeros((3,2))
# for on the instances
for j,d in enumerate(datas):
    # for on the trials
    curr_perf = np.zeros(N_TRIALS)
    for i in range(N_TRIALS):
        params = np.array(d[i]["last_param"])
        if MULTI_LINEAR:
            params = params.reshape(env.action_dim,env.state_dim)
        pol.parameters = copy.deepcopy(params)
        worker_dict = dict(
            env=copy.deepcopy(env),
            pol=copy.deepcopy(pol),
            dp=copy.deepcopy(dp),
            params=None,
            starting_state=None,
            learn_std=0,
            e_parameterization_score=0
        )

        # build the parallel functions
        delayed_functions = delayed(pg_sampling_worker)

        # parallel computation
        res = Parallel(n_jobs=N_J, backend="loky")(
            delayed_functions(**worker_dict) for _ in range(N)
        )

        # Update performance
        perf_vector = np.zeros(N, dtype=np.float64)
        for z in range(N):
            perf_vector[z] = res[z][TrajectoryResults2.PERF]

        curr_perf[i] = np.mean(perf_vector)
    extracted_p_perf[j,0] = np.mean(curr_perf)
    extracted_p_perf[j,1] = np.std(curr_perf)
print(extracted_p_perf)
np.save(f"/Users/ale/code/PyProjects/results/extraction/{ENV}_p.npy", extracted_p_perf)


# file opening

if ENV == "swimmer":
    NAME_ENV = "swimmer2"
else:
    NAME_ENV = ENV
    
if SSD:
    base_name = "/Volumes/FastAle/pes_pel/pes/"
else:
    base_name = f"/Users/ale/code/PyProjects/results/{NAME_ENV}/"


base_name += f"pes_exp_{EXP_S}_phases_{PHASES}_lrsigma_adam_{LR_SIGMA_S_A}_pg_{INNER_LOOP}_{ENV}_{HORIZON}_adam_{INNER_LR_A}_gaussian_batch_100_noclip_{PARAMS}_var_1/"
data = []
for i in range(N_TRIALS):
    name = base_name + f"pes_exp_{EXP_S}_phases_{PHASES}_lrsigma_adam_{LR_SIGMA_S_A}_pg_{INNER_LOOP}_{ENV}_{HORIZON}_adam_{INNER_LR_A}_gaussian_batch_100_noclip_{PARAMS}_var_1_trial_{i}/"
    with open(name + "apes_results.json", "r") as f:
        data.append(deepcopy(json.load(f)))

if SSD:
    base_name = "/Volumes/FastAle/pes_pel/pes/"
else:
    base_name = f"/Users/ale/code/PyProjects/results/{NAME_ENV}/"
base_name += f"pel_param_exp_exp_{EXP_L_A}_phases_{PHASES*INNER_LOOP}_lrsigma_adam_{LR_SIGMA_L_A}_pg_1_{ENV}_{HORIZON}_adam_{INNER_LR_A}_gaussian_batch_100_noclip_{PARAMS}_var_1/"
data_pel = []
for i in range(N_TRIALS):
    name = base_name + f"pel_param_exp_exp_{EXP_L_A}_phases_{PHASES*INNER_LOOP}_lrsigma_adam_{LR_SIGMA_L_A}_pg_1_{ENV}_{HORIZON}_adam_{INNER_LR_A}_gaussian_batch_100_noclip_{PARAMS}_var_1_trial_{i}/"
    with open(name + "apel_results.json", "r") as f:
        data_pel.append(deepcopy(json.load(f)))

# if SSD:
#     base_name = "/Volumes/FastAle/pes_pel/pes/"
# else:
#     base_name = f"/Users/ale/code/PyProjects/results/{ENV}/"
# base_name += f"pel_param_exp_exp_{EXP_L}_phases_{PHASES*INNER_LOOP}_lrsigma_adam_{LR_SIGMA_L_A}_pg_1_{ENV}_{HORIZON}_adam_{INNER_LR_A}_gaussian_batch_100_noclip_{PARAMS}_var_1/"
# data_pel = []
# for i in range(N_TRIALS):
#     name = base_name + f"pel_param_exp_exp_{EXP_L}_phases_{PHASES*INNER_LOOP}_lrsigma_adam_{LR_SIGMA_L_A}_pg_1_{ENV}_{HORIZON}_adam_{INNER_LR_A}_gaussian_batch_100_noclip_{PARAMS}_var_1_trial_{i}/"
#     with open(name + "apel_results.json", "r") as f:
#         data_pel.append(deepcopy(json.load(f)))

# urgent opening
if URGENT:
    au_data = []
    for i in range(N_TRIALS):
        name = path + a_path + "/" + a_path + f"_trial_{i}/"
        with open(name + "apes_results.json", "r") as f:
            au_data.append(deepcopy(json.load(f)))

datas = [data, data_pel, au_data]
extracted_a_perf = np.zeros((3,2))
# for on the instances
for j,d in enumerate(datas):
    # for on the trials
    curr_perf = np.zeros(N_TRIALS)
    for i in range(N_TRIALS):
        params = np.array(d[i]["last_param"])
        if MULTI_LINEAR:
            params = params.reshape(env.action_dim,env.state_dim)
        pol.parameters = copy.deepcopy(params)
        worker_dict = dict(
            env=copy.deepcopy(env),
            pol=copy.deepcopy(pol),
            dp=copy.deepcopy(dp),
            params=None,
            starting_state=None,
            learn_std=0,
            e_parameterization_score=0
        )

        # build the parallel functions
        delayed_functions = delayed(pg_sampling_worker)

        # parallel computation
        res = Parallel(n_jobs=N_J, backend="loky")(
            delayed_functions(**worker_dict) for _ in range(N)
        )

        # Update performance
        perf_vector = np.zeros(N, dtype=np.float64)
        for z in range(N):
            perf_vector[z] = res[z][TrajectoryResults2.PERF]

        curr_perf[i] = np.mean(perf_vector)
    extracted_a_perf[j,0] = np.mean(curr_perf)
    extracted_a_perf[j,1] = np.std(curr_perf)
print(extracted_a_perf)
np.save(f"/Users/ale/code/PyProjects/results/extraction/{ENV}_a.npy", extracted_a_perf)