# Libraries
import numpy as np
import matplotlib.pyplot as plt
import json
from algorithms.samplers import *
from envs import *
from policies import *
from data_processors import *
from tqdm import tqdm

# Globals
N_JOBS = 8
N_TRAJECTORIES = 100
ITE = 500

WHAT = "pg"
POL = "nn"

if WHAT == "pgpe":
    """path = [
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_linear_policy_16_var_1",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_linear_policy_16_var_10",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_linear_policy_16_var_01",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_linear_policy_16_var_100",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_linear_policy_16_var_001",
    ]"""
    path = [
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_nn_policy_50_var_001",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_nn_policy_50_var_01",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_nn_policy_50_var_1",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_nn_policy_50_var_10",
        "/Users/ale/results/pgpe/pgpe_test_500_adam_01_swimmer_200_nn_policy_50_var_100"
    ]
else:
    """path = [
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_lingauss_policy_16_var_1",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_lingauss_policy_16_var_10",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_lingauss_policy_16_var_01",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_lingauss_policy_16_var_100",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_lingauss_policy_16_var_001",
    ]"""
    path = [
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_nn_policy_50_var_001",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_nn_policy_50_var_01",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_nn_policy_50_var_1",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_nn_policy_50_var_10",
        "/Users/ale/results/pg/pg_test_500_adam_001_swimmer_200_nn_policy_50_var_100"
    ]

env = Swimmer(horizon=200, gamma=0.995, render=False)
dp = IdentityDataProcessor()

if POL != "nn":
    pol = LinearPolicy(parameters=np.ones(16), dim_state=8, dim_action=2, multi_linear=True)
else:
    net = nn.Sequential(
        nn.Linear(env.state_dim, 5, bias=False),
        nn.Linear(5, env.action_dim, bias=False)
    )
    model_desc = dict(
        layers_shape=[(env.state_dim, 5), (5, env.action_dim)]
    )
    pol = NeuralNetworkPolicy(
        parameters=None,
        input_size=env.state_dim,
        output_size=env.action_dim,
        model=copy.deepcopy(net),
        model_desc=copy.deepcopy(model_desc)
    )


# Function to sample a the deterministic policy curves
def sample_deterministic_curve(
        env: BaseEnv = None,
        pol: BasePolicy = None,
        n_jobs: int = 1,
        n_trajectories: int = 1,
        ite: int = 100,
        parameter_schedule: np.array = None
) -> np.array:
    # Checks
    assert len(parameter_schedule) == ite
    assert n_jobs == -1 or n_jobs > 0
    assert n_trajectories >= 1

    # Collect
    mean_perf = np.zeros(ite, dtype=np.float128)
    for i in tqdm(range(ite)):
        pol.set_parameters(thetas=parameter_schedule[i, :])
        worker_dict = dict(
            env=copy.deepcopy(env),
            pol=copy.deepcopy(pol),
            dp=IdentityDataProcessor(),
            params=copy.deepcopy(parameter_schedule[i, :]),
            starting_state=None
        )
        # build the parallel functions
        delayed_functions = delayed(pg_sampling_worker)

        # parallel computation
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed_functions(**worker_dict) for _ in range(n_trajectories)
        )

        # extract data
        ite_perf = np.zeros(n_trajectories, dtype=np.float128)
        for j in range(n_trajectories):
            ite_perf[j] = res[j][TrajectoryResults.PERF]

        # compute mean
        mean_perf[i] = np.mean(ite_perf)

    return mean_perf


if __name__ == "__main__":
    for p in path:
        print("Reading: " + p)

        # Open the file
        name = p + f"/{WHAT}_results.json"
        file = open(name)
        data = json.load(file)

        # extract data
        p_history = None
        if WHAT == "pgpe":
            p_history = np.array(data["rho_history"])
        else:
            p_history = np.array(data["thetas_history"])
        # compute the deterministic perf
        deterministic_res = sample_deterministic_curve(
            env=env,
            pol=pol,
            parameter_schedule=p_history,
            n_jobs=N_JOBS,
            n_trajectories=N_JOBS,
            ite=ITE
        )

        # save the additional field
        data["performance_det"] = np.array(deterministic_res, dtype=float).tolist()
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=4))
            f.close()
