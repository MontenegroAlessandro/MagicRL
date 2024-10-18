"""
Implementation of the AD-PGPD algorithm from Ding et. al 2024
"""

from copy import deepcopy
import io
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import json

import matplotlib.pyplot as plt

from policies import BasePolicy, LinearPolicy, OldLinearPolicy, LinearGaussianPolicy
from envs import BaseEnv, CostSwimmer
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms import  BasePG_PD

class V_RES:
    Vr = 0
    Vg = 1

class Q_RES:
    s = 0
    a =1
    Qr = 2
    Qg = 3


def q_estimation(env: BaseEnv, pol: BasePolicy, dp: BaseProcessor):
    # structures
    Qr = 0
    Qg = 0
    gamma = env.gamma
    end = bool(np.random.uniform(0, 1) >= gamma)

    env.reset()

    # reset the environment (which sets also the initial state)
    start_state = deepcopy(env.state)
    start_action = pol.draw_action(start_state)

    # sample state and action
    state = deepcopy(env.state)
    action = deepcopy(start_action)

    while not end:
        feat = dp.transform(state=deepcopy(state))
        # simulation
        state, rew, _, info = env.step(action=deepcopy(action))
        cost = - info["costs"][0]
        # save
        Qr += rew
        Qg += cost

        action = pol.draw_action(state=deepcopy(feat))

        # update
        end = bool(np.random.uniform(0, 1) >= np.sqrt(gamma))

    return [start_state, start_action, Qr, Qg]

def value_estimation(env: BaseEnv, pol: BasePolicy, dp: BaseProcessor, state):
    # structures
    perf_r = 0
    perf_g = 0
    gamma = env.gamma
    end = bool(np.random.uniform(0, 1) >= gamma)

    # reset the environment (which sets also the initial state)
    if state is None:
        env.reset()
    else:
        env.reset(state=deepcopy(state))

    # sample state and action
    state = deepcopy(env.state)
    while not end:
        feat = dp.transform(state=deepcopy(state))
        # simulation
        action = pol.draw_action(state=deepcopy(feat))
        state, rew, _, info = env.step(action=deepcopy(action))
        cost = - info["costs"][0]
        # save
        perf_r += rew
        perf_g += cost
        # update
        end = bool(np.random.uniform(0, 1) >= np.sqrt(gamma))

    return [perf_r, perf_g]

class ADPGPD(BasePG_PD):
    # Note that it is thought for the Tabular Softmax
    # Note that this algorithm has the constraints as Vg >= b (with g in [0,1])
    # What we can do is to conver the cost and the threshold to be negative
    def __init__(
        self,
        ite: int = 100,
        batch: int = 100,
        pol: BasePolicy = None,
        env: BaseEnv = None,
        lr: np.ndarray = None,
        lr_strategy: str = "constant",
        dp: BaseProcessor = None,
        threshold: float = 0,
        checkpoint_freq: int = 1000,
        n_jobs: int = 1,
        directory: str = None,
        inner_loop_param: float = 100,
        reg: float = 0,
        inner_batch: int = 100
    ) -> None:
        # super class initialization
        super().__init__(
            ite=ite, env=env, pol=pol, dp=dp, batch=batch, lr=lr,
            lr_strategy=lr_strategy, threshold=threshold,
            checkpoint_freq=checkpoint_freq, n_jobs=n_jobs, directory=directory
        )

        # additional fields
        assert inner_loop_param > 0
        self.inner_loop_param = inner_loop_param

        assert reg >= 0
        self.reg = reg
        self.obj_name = "ADPGPD"

        err_msg = f"[{self.obj_name}]  adam lr strategy not yet implemented."
        if lr_strategy == "adam":
            raise NotImplementedError(err_msg)

        # structures: running
        # policy parameters

        self.theta = np.zeros(self.pol.tot_params, dtype=np.float64)
        # lag multipliers
        self.lam = 0

        # structures: history
        self.thetas = np.zeros((self.ite, self.pol.tot_params), dtype=np.float64)
        self.lambdas = np.zeros(self.ite, dtype=np.float64)
        self.cost_values = np.zeros(self.ite, dtype=np.float64)
        self.omegas = np.zeros((self.ite, self.pol.tot_params), dtype=np.float64)
        self.values = np.zeros(self.ite, dtype=np.float64)

        # additional parameters
        self.gamma = self.env.gamma
        assert self.gamma < 1, f"[{self.obj_name}] gamma must be less than one."

        # cast the problem to cost minimization
        self.threshold = - self.threshold
        assert inner_batch > 0
        self.inner_batch = inner_batch

        # best performance
        self.best_perf = - np.inf
        self.best_theta = np.zeros(self.pol.tot_params, dtype=np.float64)

    def learn(self):
        for t in tqdm(range(self.ite)):
            # save the theta history and lambda history
            self.thetas[t, :] = deepcopy(self.theta)
            self.lambdas[t] = self.lam

            # set the policy parameters
            self.pol.set_parameters(thetas=deepcopy(self.theta))

            # init to zero the omega values
            self.omega_r = np.zeros(self.pol.tot_params, dtype=np.float64)
            self.omega_r_batch = np.zeros((self.inner_batch, self.pol.tot_params), dtype=np.float64)

            # unbiased estimaiton of Ar and Ag
            q_dict = dict(
                env=deepcopy(self.env),
                pol=deepcopy(self.pol),
                dp=deepcopy(self.dp)
            )

            delayed_functions = delayed(q_estimation)
            adv_res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**q_dict) for _ in range(self.inner_batch)
            )

            for k in range(self.inner_batch):
                # sample q value
                # s, a, Ar, Ag = self._A_unbiased_estimation()
                s = adv_res[k][Q_RES.s]
                a = adv_res[k][Q_RES.a]
                Qr = adv_res[k][Q_RES.Qr]

                # set the lr
                alpha = 2 / (self.inner_loop_param * (k + 1))

                # update the actor
                # phi is defined as a column vector built repeating the state "dim_action" times over the same dimension
                id = np.eye(self.pol.tot_params)

                phi = np.array(np.tile(deepcopy(s), self.pol.dim_action))
                term1 = phi.T @  self.omega_r
                term2 = (s @ self.pol.parameters.T) @ a

                self.omega_r = self.omega_r - 2 * alpha * (term1 - Qr - term2 * self.lr_theta) * phi

                # save vectors
                self.omega_r_batch[k, :] = deepcopy(self.omega_r)

            # unbiased estimation of Vr and Vg
            rews = np.zeros(self.batch, dtype=np.float64)
            costs = np.zeros(self.batch, dtype=np.float64)

            value_dict = dict(
                pol=deepcopy(self.pol),
                env=deepcopy(self.env),
                dp=deepcopy(self.dp),
                state=None
            )

            delayed_functions = delayed(value_estimation)
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**value_dict) for _ in range(self.batch)
            )

            for i in range(self.batch):
                rews[i] = res[i][V_RES.Vr]
                costs[i] = res[i][V_RES.Vg]
                # note that the costs are already "-" in the function!
            self.values[t] = np.mean(rews)
            self.cost_values[t] = np.mean(costs)

            # compute the final omega for the iteration
            k_series = (np.arange(self.inner_batch, dtype=np.float64) + 1)[:,np.newaxis]
            omega_hat_r = (2 / (self.inner_batch * (self.inner_batch + 1))) * np.sum(k_series * self.omega_r_batch, axis=0)

            self.omegas[t, :] = deepcopy(omega_hat_r)

            # update parameters
            if self.lr_strategy == "constant":
                # primal update

                # build an array with all the acted actions
                adv_selected = np.array([adv_res[i][Q_RES.a] for i in range(len(adv_res))])

                # compute the norm of the actions
                adv_norm = np.linalg.norm(adv_selected, axis=0)

                # update the parameter vector of the policy
                self.theta = omega_hat_r - np.mean((self.reg + self.lr_theta) * adv_norm, axis=0)

                # dual update
                self.lam = np.clip(self.lam - self.lr_lambda*(self.cost_values[t] - self.threshold + self.reg * self.lam), 0, np.inf)
            else:
                # adam lr strategy (NOT YET IMPLEMENTED)
                pass

            if self.values[t] > self.best_perf:
                print(f'\n{self.obj_name} - New best performance found at iteration {t}')
                print(f"[{self.obj_name}] mean trajectory reward:\t {self.values[t]}")
                print(f"[{self.obj_name}] mean trajectory cost:\t {self.cost_values[t]}")
                print(f"[{self.obj_name}] lambda:\t {self.lam}")
                print(f"[{self.obj_name}] theta:\t {self.theta}")
                self.best_perf = self.values[t]
                self.best_theta = deepcopy(self.theta)

            # save
            if not (t % self.checkpoint_freq):
                self.save_results()

    def save_results(self):
        """Save the results."""
        results = {
            "values": np.array(self.values, dtype=float).tolist(),
            "omegas": np.array(self.omegas, dtype=float).tolist(),
            "costs": np.array(self.cost_values, dtype=float).tolist(),
            "theta_history": np.array(self.thetas, dtype=float).tolist(),
            "lambda_history": np.array(self.lambdas, dtype=float).tolist(),
            "best_theta": np.array(self.best_theta, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + f"/{self.obj_name}_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return

def train():
    ite = 301
    gamma = 0.99
    batch = 100
    env_class = CostSwimmer
    horizon = 100
    clip = 1
    reg = 0.0001
    lr = [0.001, 0.01]
    lr_strategy = "constant"
    n_workers = 6
    directory  ='/Users/leonardo/Desktop/Thesis/Data'
    b = 100

    env = env_class(horizon=horizon, gamma=gamma, render=False, clip=bool(clip))

    s_dim = env.state_dim
    a_dim = env.action_dim
    tot_params = s_dim * a_dim
    dp = IdentityDataProcessor(dim_feat=env.state_dim)

    policy = LinearGaussianPolicy(
        parameters=np.zeros(tot_params),
        dim_state=s_dim,
        dim_action=a_dim,
        multi_linear=True
    )

    alg_parameters = dict(
        lr=lr,
        lr_strategy=lr_strategy,
        batch = batch,
        ite=ite,
        env=env,
        pol=policy,
        dp=dp,
        n_jobs=n_workers,
        reg = reg,
        directory=directory,
        threshold = b,
        checkpoint_freq = 100
    )
    alg = ADPGPD(**alg_parameters)

    print("START LEARNING")

    alg.learn()

def plot():

    with open('/Users/leonardo/Desktop/Thesis/Data/ADPGPD_results.json') as f:
        data = json.load(f)

    # pot the "values" column of data
    plt.plot(data['values'])

    plt.show()


if __name__ == "__main__":
    train()

