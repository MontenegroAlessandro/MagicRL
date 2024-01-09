# Libraries
import argparse
from algorithms import *
from data_processors import *
from envs import *
from policies import *
from art import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--server",
    help="Location.",
    type=int,
    default=1
)
parser.add_argument(
    "--ite",
    help="How many iterations the algorithm must do.",
    type=int,
    default=100
)
parser.add_argument(
    "--alg",
    help="The algorithm to use.",
    type=str,
    default="pgpe",
    choices=["pgpe", "pg"]
)
parser.add_argument(
    "--var",
    help="The exploration amount.",
    type=float,
    default=1
)
parser.add_argument(
    "--pol",
    help="The policy used.",
    type=str,
    default="linear",
    choices=["linear", "nn"]
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="swimmer",
    choices=["swimmer", "half_cheetah"]
)
parser.add_argument(
    "--horizon",
    help="The horizon amount.",
    type=int,
    default=100
)
parser.add_argument(
    "--gamma",
    help="The gamma amount.",
    type=float,
    default=1
)
parser.add_argument(
    "--lr",
    help="The lr amount.",
    type=float,
    default=1e-3
)
parser.add_argument(
    "--lr_strategy",
    help="The strategy employed for the lr.",
    type=str,
    default="adam",
    choices=["adam", "constant"]
)
parser.add_argument(
    "--n_workers",
    help="How many parallel cores.",
    type=int,
    default=1
)
parser.add_argument(
    "--batch",
    help="The batch size.",
    type=int,
    default=100
)

args = parser.parse_args()

# Preprocess Arguments
if args.alg == "pg":
    if args.pol == "linear":
        args.pol = "gaussian"
    elif args.pol == "nn":
        args.pol = "deep_gaussian"

# Build
if args.server:
    dir_name = f"/data/alessandro/{args.alg}/"
else:
    dir_name = f"/Users/ale/results/{args.alg}/"
dir_name += f"{args.alg}_{args.ite}_{args.env}_{args.horizon}_{args.lr_strategy}_"
dir_name += f"{str(args.lr).replace('.', '')}_{args.pol}_"

"""Environment"""
MULTI_LINEAR = False
if args.env == "swimmer":
    env_class = Swimmer
    env = Swimmer(horizon=args.horizon, gamma=args.gamma, render=False)
    MULTI_LINEAR = True
elif args.env == "half_cheetah":
    env_class = HalfCheetah
    env = HalfCheetah(horizon=args.horizon, gamma=args.gamma, render=False)
    MULTI_LINEAR = True
else:
    raise ValueError(f"Invalid env name.")
s_dim = env.state_dim
a_dim = env.action_dim

"""Data Processor"""
dp = IdentityDataProcessor()

"""Policy"""
if args.pol == "linear":
    tot_params = s_dim * a_dim
    pol = LinearPolicy(
        parameters=np.ones(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        multi_linear=MULTI_LINEAR
    )
elif args.pol == "gaussian":
    tot_params = s_dim * a_dim
    pol = LinearGaussianPolicy(
        parameters=np.ones(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        std_dev=np.sqrt(args.var),
        std_decay=0,
        std_min=1e-6,
        multi_linear=MULTI_LINEAR
    )
elif args.pol in ["nn", "deep_gaussian"]:
    net = nn.Sequential(
        nn.Linear(s_dim, 16, bias=False),
        nn.Tanh(),
        nn.Linear(16, a_dim, bias=False)
    )
    model_desc = dict(
        layers_shape=[(s_dim, 16), (16, a_dim)]
    )
    if args.pol == "nn":
        pol = NeuralNetworkPolicy(
            parameters=None,
            input_size=s_dim,
            output_size=a_dim,
            model=copy.deepcopy(net),
            model_desc=copy.deepcopy(model_desc)
        )
    elif args.pol == "deep_gaussian":
        pol = DeepGaussian(
            parameters=None,
            input_size=s_dim,
            output_size=a_dim,
            model=copy.deepcopy(net),
            model_desc=copy.deepcopy(model_desc),
            std_dev=np.sqrt(args.var),
            std_decay=0,
            std_min=1e-6
        )
    else:
        raise ValueError("Invalid nn policy name.")
    tot_params = pol.tot_params
else:
    raise ValueError(f"Invalid policy name.")
dir_name += f"{tot_params}_var_{str(args.var).replace('.','')}"

"""Algorithm"""
if args.alg == "pgpe":
    hp = np.zeros((2, tot_params))
    hp[0] = [0] * tot_params
    hp[1] = [np.log(np.sqrt(args.var))] * tot_params
    alg_parameters = dict(
        lr=[args.lr],
        initial_rho=hp,
        ite=args.ite,
        batch_size=args.batch,
        episodes_per_theta=1,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir_name,
        verbose=False,
        natural=False,
        checkpoint_freq=100,
        lr_strategy=args.lr_strategy,
        learn_std=False,
        std_decay=0,
        std_min=1e-6,
        n_jobs_param=args.n_workers,
        n_jobs_traj=1
    )
    alg = PGPE(**alg_parameters)
elif args.alg == "pg":
    alg_parameters = dict(
        lr=[args.lr],
        lr_strategy=args.lr_strategy,
        estimator_type="GPOMDP",
        initial_theta=[0] * tot_params,
        ite=args.ite,
        batch_size=args.batch,
        env=env,
        policy=pol,
        data_processor=dp,
        directory=dir_name,
        verbose=False,
        natural=False,
        checkpoint_freq=100,
        n_jobs=args.n_workers
    )
    alg = PolicyGradient(**alg_parameters)
else:
    raise ValueError("Invalid algorithm name.")

if __name__ == "__main__":
    print(text2art(f"== {args.alg} TEST on {args.env} =="))
    print(args)
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    print(alg.performance_idx)
