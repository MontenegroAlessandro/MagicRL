# Libraries
import argparse
from algorithms import CPGPE, CPolicyGradient, NaturalPG_PD, NaturalPG_PD_2
from data_processors import IdentityDataProcessor, GWTabularProcessor, LQRTabularProcessor, GWDataProcessorRBF
from data_processors.robot_world_processor import RobotWorldProcessor
from envs import *
from policies import *
from art import *
from algorithms.utils import LearnRates

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
    default="cpgpe",
    choices=["cpgpe", "cpg", "npgpd", "rpgpd", "npgpd2", "rpgpd2"]
)
parser.add_argument(
    "--risk",
    help="The risk measure to use.",
    type=str,
    default="tc",
    choices=["tc", "mv", "cvar", "chance"]
)
parser.add_argument(
    "--reg",
    help="The regularization amount.",
    type=float,
    default=0
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
    choices=["linear", "nn", "big_nn", "softmax", "gw_pol", "gaussian_rbf"]
)
parser.add_argument(
    "--env",
    help="The environment.",
    type=str,
    default="lqr",
    choices=["lqr", "gw_d", "gw_c", "swimmer", "hopper", "half_cheetah", "robot_world"]
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
    nargs="+",
    type=float,
    required=True
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
    "--n_trials",
    help="How many runs of the same experiment to perform.",
    type=int,
    default=1
)
parser.add_argument(
    "--env_param",
    help="Additional parameter for the environment.",
    type=int,
    default=1
)
parser.add_argument(
    "--risk_param",
    help="Additional parameter for the risk.",
    type=float,
    nargs="+"
)
parser.add_argument(
    "--c_bounds",
    help="Thresholds for the costs.",
    type=float,
    nargs="+",
    required=True
)
parser.add_argument(
    "--l_init",
    help="Initial values for the lambdas.",
    type=float,
    nargs="+"
)
parser.add_argument(
    "--eta_init",
    help="Initial values for the etas.",
    type=float,
    nargs="+"
)
parser.add_argument(
    "--alternate",
    help="Use alternate gradient ascent descent.",
    type=int,
    default=1,
    choices=[0, 1]
)
parser.add_argument(
    "--deterministic",
    help="Enable the sampling of the deterministic performance.",
    type=int,
    default=1,
    choices=[0, 1]
)
parser.add_argument(
    "--clip",
    help="Whether to clip the action in the environment.",
    type=int,
    default=0,
    choices=[0, 1]
)

args = parser.parse_args()

if args.alg != "cpgpe":
    if args.pol == "linear":
        args.pol = "gaussian"
    elif args.pol == "nn":
        args.pol = "deep_gaussian"

if args.var < 1:
    string_var = str(args.var).replace(".", "")
else:
    string_var = str(int(args.var))

if 0 < args.reg < 1:
    string_reg = str(args.reg).replace(".", "")
else:
    string_reg = str(int(args.reg))

# Build
base_dir = args.dir

for i in range(args.n_trials):
    np.random.seed(i)
    dir_name = f"{args.alg}_{args.ite}_{args.env}_{args.horizon}_{args.lr_strategy}_"
    dir_name += f"p{str(args.lr[LearnRates.PARAM]).replace('.', '')}_d{str(args.lr[LearnRates.LAMBDA]).replace('.', '')}_{args.pol}_batch_{args.batch}_"
    dir_name += f"reg_{string_reg}_risk_{args.risk}_"

    """Environment"""
    MULTI_LINEAR = False
    if args.env == "lqr":
        if args.alg in ["cpgpe", "cpg", "npgpd2", "rpgpd2"]:
            env = CostLQR.generate(
                s_dim=args.env_param,
                a_dim=args.env_param,
                horizon=args.horizon,
                gamma=args.gamma,
                scale_matrix=0.9,
                max_pos=np.inf,
                max_action=np.inf
            )
            MULTI_LINEAR = bool(args.env_param > 1)
        else:
            env = CostLQRDiscrete.generate(
                s_dim=args.env_param,
                a_dim=args.env_param,
                horizon=args.horizon,
                gamma=args.gamma,
                scale_matrix=0.9,
                state_bins=5,
                action_bins=5,
                max_pos=10,
                max_action=10
            )
            MULTI_LINEAR = bool(args.env_param > 1)
    elif args.env == "swimmer":
        env = CostSwimmer(
            horizon=args.horizon,
            gamma=args.gamma,
            verbose=False,
            render=False,
            clip=args.clip
        )
        MULTI_LINEAR = True
    elif args.env == "hopper":
        env = CostHopper(
            horizon=args.horizon,
            gamma=args.gamma,
            verbose=False,
            render=False,
            clip=args.clip
        )
        MULTI_LINEAR = True
    elif args.env == "half_cheetah":
        env = CostHalfCheetah(
            horizon=args.horizon,
            gamma=args.gamma,
            verbose=False,
            render=False,
            clip=args.clip
        )
        MULTI_LINEAR = True
    elif args.env == "gw_d":
        env = GridWorldEnvDisc(
            horizon=args.horizon,
            gamma=args.gamma,
            grid_size=7,
            reward_type="linear",
            env_type="U",
            render=False,
            dir=None,
            random_init=True,
            ding_flag=False
        )
    elif args.env == "gw_c":
        env = GridWorldEnvCont(
            horizon=args.horizon,
            gamma=args.gamma,
            grid_size=7,
            reward_type="linear",
            use_costs = True,
            render = False,
            threshold=0.0,
            sampling_args  = { "n_samples": 1, "radius": 0.1},
            # dir = "/Users/leonardo/Desktop/Thesis/Data/GridWorld"
        )
    elif args.env == "robot_world":
        env = RobotWorld.generate(
            s_dim = 2 * args.env_param,
            a_dim = args.env_param,
            horizon=args.horizon,
            gamma=args.gamma,
            max_pos=10,
            max_action=10,
            scale_matrix=0.9
        )
        MULTI_LINEAR = True
    else:
        raise ValueError(f"Invalid env name.")
    s_dim = env.state_dim
    a_dim = env.action_dim

    """Data Processor"""
    basis = 5
    if args.env == "gw_d":
        dp = GWTabularProcessor(index_map=env.grid_index)
    elif args.env == "lqr" and args.alg in ["rpgpd", "npgpd"]:
        dp = LQRTabularProcessor(index_map=env.state_enumeration)
    elif args.env == "gw_c":
        dp = GWDataProcessorRBF(num_basis=basis, grid_size=7, std_dev=0.75) # basis = 7, std_dev = 0.6
    elif args.env == "robot_world":
        dp =  RobotWorldProcessor()
        s_dim = 1 + 3 * s_dim
    else:
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
        raise NotImplementedError
    elif args.pol == "softmax":
        tot_params = env.state_dim * env.action_dim
        temperature = 1 if args.alg != "cpgpe" else 0.1
        pol = TabularSoftmax(
            dim_state=env.state_dim,
            dim_action=env.action_dim,
            tot_params=tot_params,
            temperature=temperature,
            deterministic=bool(args.alg == "cpgpe")
        )
    elif args.pol == "gw_pol":
        tot_params = basis * s_dim
        pol = GWPolicy(
            dim_state=env.state_dim,
            thetas=np.zeros(tot_params).tolist(),
            std_dev=np.sqrt(args.var),
            alg=args.alg
        )
    else:
        raise ValueError(f"Invalid policy name.")
    # dir_name += f"{tot_params}_var_{string_var}_trial_{i}"
    if bool(args.alternate):
        dir_name += f"p{tot_params}_var_{string_var}_a"
    else:
        dir_name += f"p{tot_params}_var_{string_var}_na"
    dir_name = base_dir + dir_name + "/" + dir_name + f"_trial_{i}"

    """Algorithm"""
    if args.alg == "cpgpe":
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
            cost_type=args.risk,
            cost_param=np.array(args.risk_param),
            omega=args.reg,
            thresholds=np.array(args.c_bounds),
            lambda_init=np.array(args.l_init),
            eta_init=np.array(args.eta_init),
            alternate=bool(args.alternate),
            lr=args.lr,
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
            checkpoint_freq=250,
            lr_strategy=args.lr_strategy,
            learn_std=False,
            std_decay=0,
            std_min=1e-6,
            n_jobs_param=args.n_workers,
            n_jobs_traj=1,
            deterministic=bool(args.deterministic)
        )
        alg = CPGPE(**alg_parameters)
    elif args.alg == "cpg":
        if args.pol in ["gaussian", "softmax"]:
            init_theta = [0] * tot_params
        else:
            init_theta = np.random.normal(0, 1, tot_params)
        alg_parameters = dict(
            cost_type=args.risk,
            cost_param=np.array(args.risk_param),
            omega=args.reg,
            thresholds=np.array(args.c_bounds),
            lambda_init=np.array(args.l_init),
            eta_init=np.array(args.eta_init),
            alternate=bool(args.alternate),
            lr=args.lr,
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
            checkpoint_freq=250,
            n_jobs=args.n_workers,
            deterministic=bool(args.deterministic)
        )
        alg = CPolicyGradient(**alg_parameters)
    elif args.alg == "npgpd":
        alg_parameters = dict(
            directory=dir_name,
            ite=args.ite,
            batch=args.batch,
            pol=pol,
            env=env,
            lr=args.lr[:2],
            lr_strategy=args.lr_strategy,
            dp=dp,
            threshold=args.c_bounds[0],
            n_jobs=args.n_workers,
            reg=0
        )
        alg = NaturalPG_PD(**alg_parameters)
    elif args.alg == "rpgpd":
        alg_parameters = dict(
            directory=dir_name,
            ite=args.ite,
            batch=args.batch,
            pol=pol,
            env=env,
            lr=args.lr[:2],
            lr_strategy=args.lr_strategy,
            dp=dp,
            threshold=args.c_bounds[0],
            n_jobs=args.n_workers,
            reg=args.reg
        )
        alg = NaturalPG_PD(**alg_parameters)
        """alg_parameters = dict(
            directory=dir_name,
            ite=args.ite,
            batch=args.batch,
            pol=pol,
            env=env,
            lr=args.lr[:2],
            lr_strategy=args.lr_strategy,
            dp=dp,
            threshold=args.c_bounds[0],
            n_jobs=args.n_workers,
            reg=args.reg,
            inner_loop_param=1
        )
        alg = RegularizedPG_PD(**alg_parameters)"""
    elif args.alg in ["npgpd2", "rpgpd2"]:
        alg_parameters = dict(
            directory=dir_name,
            ite=args.ite,
            batch=args.batch,
            pol=pol,
            env=env,
            lr=args.lr[:2],
            lr_strategy=args.lr_strategy,
            dp=dp,
            threshold=args.c_bounds[0],
            n_jobs=args.n_workers,
            reg=args.reg,
            inner_loop_param=100000,
            inner_batch=args.batch * 5
        )
        alg = NaturalPG_PD_2(**alg_parameters)
    else:
        raise ValueError("Invalid algorithm name.")

    print(text2art(f"== {args.alg} TEST on {args.env} =="))
    print(text2art(f"Trial {i}"))
    print(args)
    print(text2art("Learn Start"))
    alg.learn()
    alg.save_results()
    if args.alg in ["cpgpe", "cpg"]:
        print(alg.performance_idx)
        print(alg.risk_idx)
