# Libraries
import argparse
from algorithms import PGPE, PolicyGradient, DeterministicPG, CPGPE
from data_processors import IdentityDataProcessor
from envs import *
from policies import *
from art import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--dir",
    help="Directory in which save the results.",
    type=str,
    default=""
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
    choices=["pgpe", "pg", "dpg", "cpgpe"]
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
    choices=["linear", "nn", "big_nn"]
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="swimmer",
    choices=["swimmer", "half_cheetah", "reacher", "humanoid", "ant", "hopper", "lqr"]
)
parser.add_argument(
    "--costs",
    help="Flag to ensure the usage of the costly-version of the environment.",
    type=int,
    default=0,
    choices=[0, 1]
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
parser.add_argument(
    "--clip",
    help="Whether to clip the action in the environment.",
    type=int,
    default=1,
    choices=[0, 1]
)
parser.add_argument(
    "--n_trials",
    help="How many runs of the same experiment to perform.",
    type=int,
    default=1
)
parser.add_argument(
    "--lqr_state_dim",
    help="State dimension for the LQR environment.",
    type=int,
    default=1
)
parser.add_argument(
    "--lqr_action_dim",
    help="Action dimension for the LQR environment.",
    type=int,
    default=2
)

args = parser.parse_args()

huge = False
if args.pol == "big_nn":
    huge = True
    args.pol = "nn"

if args.alg == "pg":
    if args.pol == "linear":
        args.pol = "gaussian"
    elif args.pol == "nn":
        args.pol = "deep_gaussian"

if args.var < 1:
    string_var = str(args.var).replace(".", "")
else:
    string_var = str(int(args.var))

# Build
base_dir = args.dir

for i in range(args.n_trials):
    np.random.seed(i)
    dir_name = f"{args.alg}_{args.ite}_{args.env}_{args.horizon}_{args.lr_strategy}_"
    dir_name += f"{str(args.lr).replace('.', '')}_{args.pol}_batch_{args.batch}_"
    if args.clip:
        dir_name += "clip_"
    else:
        dir_name += "noclip_"

    """Environment"""
    MULTI_LINEAR = False
    if args.env == "swimmer":
        if args.costs:
            raise NotImplementedError
        else:
            env_class = Swimmer
            env = Swimmer(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "half_cheetah":
        if args.costs:
            raise NotImplementedError
        else:
            env_class = HalfCheetah
            env = HalfCheetah(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "reacher":
        if args.costs:
            raise NotImplementedError
        else:
            env_class = Reacher
            env = Reacher(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "humanoid":
        if args.costs:
            raise NotImplementedError
        else:
            env_class = Humanoid
            env = Humanoid(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "ant":
        if args.costs:
            raise NotImplementedError
        else:
            env_class = Ant
            env = Ant(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "hopper":
        if args.costs:
            raise NotImplementedError
        else:
            env_class = Hopper
            env = Hopper(horizon=args.horizon, gamma=args.gamma, render=False, clip=bool(args.clip))
        MULTI_LINEAR = True
    elif args.env == "lqr":
        env_class = CostLQR if args.costs else LQR
        env = env_class.generate(
                s_dim=args.lqr_state_dim,
                a_dim=args.lqr_action_dim,
                horizon=args.horizon,
                gamma=args.gamma,
                scale_matrix=0.9
            )
        MULTI_LINEAR = bool(args.lqr_action_dim > 1)
    else:
        raise ValueError(f"Invalid env name.")
    s_dim = env.state_dim
    a_dim = env.action_dim

    """Data Processor"""
    dp = IdentityDataProcessor(dim_feat=env.state_dim)

    """Policy"""
    if args.pol == "linear":
        tot_params = s_dim * a_dim
        """pol = LinearPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            sigma_noise=0
        )"""
        pol = OldLinearPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            multi_linear=MULTI_LINEAR
        )
    elif args.pol == "gaussian":
        tot_params = s_dim * a_dim
        pol = LinearGaussianPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            std_dev=np.sqrt(args.var),
            std_decay=0,
            std_min=1e-5,
            multi_linear=MULTI_LINEAR
        )
        """pol = LinearPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            sigma_noise=np.sqrt(args.var),
            sigma_decay=0,
            sigma_min=1e-5
        )"""
    elif args.pol in ["nn", "deep_gaussian"]:
        if not huge:
            net = nn.Sequential(
                nn.Linear(s_dim, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, 32, bias=False),
                nn.Tanh(),
                nn.Linear(32, a_dim, bias=False)
            )
            model_desc = dict(
                layers_shape=[(s_dim, 32), (32, 32), (32, a_dim)]
            )
        else:
            net = nn.Sequential(
                nn.Linear(s_dim, 100, bias=False),
                nn.Tanh(),
                nn.Linear(100, 50, bias=False),
                nn.Tanh(),
                nn.Linear(50, 25, bias=False),
                nn.Tanh(),
                nn.Linear(25, a_dim, bias=False)
            )
            model_desc = dict(
                layers_shape=[(s_dim, 100), (100, 50), (50, 25), (25, a_dim)]
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
    # dir_name += f"{tot_params}_var_{string_var}_trial_{i}"
    dir_name += f"{tot_params}_var_{string_var}"
    dir_name = base_dir + dir_name + "/" + dir_name + f"_trial_{i}"

    """Algorithm"""
    if args.alg == "pgpe":
        if args.var == 1:
            var_term = 1.001
        else:
            var_term = args.var
        hp = np.zeros((2, tot_params))
        if args.pol == "linear":
            hp[0] = [0] * tot_params
        else:
            hp[0] = np.random.normal(0, 1, tot_params)
        hp[1] = [np.log(np.sqrt(var_term))] * tot_params
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
    elif args.alg == "cpgpe":
        if args.var == 1:
            var_term = 1.001
        else:
            var_term = args.var
        hp = np.zeros((2, tot_params))
        if args.pol == "linear":
            hp[0] = [0] * tot_params
        else:
            hp[0] = np.random.normal(0, 1, tot_params)
        hp[1] = [np.log(np.sqrt(var_term))] * tot_params
        alg_parameters = dict(
            cost_type="tc",
            cost_param=0,
            omega=.1,
            thresholds=np.array([0.5]),
            lambda_init=np.array([0]),
            eta_init=0,
            lr=[args.lr, args.lr * 10, args.lr],
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
        alg = CPGPE(**alg_parameters)
    elif args.alg == "pg":
        if args.pol == "gaussian":
            init_theta = [0] * tot_params
        else:
            init_theta = np.random.normal(0, 1, tot_params)
        alg_parameters = dict(
            lr=[args.lr],
            lr_strategy=args.lr_strategy,
            estimator_type="GPOMDP",
            initial_theta=init_theta,
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
    elif args.alg == "dpg":
        if args.pol == "linear":
            init_theta = [0] * tot_params
        else:
            init_theta = np.random.normal(0, 1, tot_params)
        pol.set_parameters(init_theta)
        b_pol = copy.deepcopy(pol)
        b_pol.sigma_noise = np.sqrt(args.var)
        alg_parameters = dict(
            ite=args.ite,
            directory=dir_name,
            det_pol=pol,
            b_pol=b_pol,
            env=env,
            batch=args.batch,
            value_features=dp,
            b_pol_features=dp,
            theta_step=args.lr,
            omega_step=args.lr*10,
            v_step=args.lr*10,
            lr_strategy=args.lr_strategy,
            checkpoint_freq=100,
            save_det_curve=True,
            n_jobs=args.n_workers,
            env_seed=i,
            update_b_pol=True
        )
        alg = DeterministicPG(**alg_parameters)
    else:
        raise ValueError("Invalid algorithm name.")

    print(text2art(f"== {args.alg} TEST on {args.env} =="))
    print(text2art(f"Trial {i}"))
    print(args)
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    if args.alg != "dpg":
        print(alg.performance_idx)
    if args.alg == "cpgpe":
        print(alg.cost_idx)
