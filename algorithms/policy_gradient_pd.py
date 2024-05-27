"""
Implmentation of NPG-PD and RPG-PD.
These algorithms only work for DISCRETE state and action spaces.
References: Ding et al. (sample-based versions with tabular softmax).
"""
# Libraries
import numpy as np
from joblib import Parallel, delayed
from data_processors.base_processor import BaseProcessor
from policies.base_policy import BasePolicy
from policies.softmax_policy import TabularSoftmax
from envs.base_env import BaseEnv
from abc import ABC, abstractmethod
from copy import deepcopy
from algorithms.samplers import pg_sampling_worker
from algorithms.utils import TrajectoryResults, LearnRates, check_directory_and_create
from tqdm import tqdm
from adam.adam import Adam
import io, json


class ADV_RES:
    s = 0
    a = 1
    Ar = 2
    Ag = 3

class V_RES:
    Vr = 0
    Vg = 1

def advantage_estimation(env: BaseEnv, pol: BasePolicy, dp: BaseProcessor):
    # structures
    perf_r = 0
    perf_g = 0
    gamma = env.gamma
    end = False
    
    ### SPOT sh AND ah ###
    # reset the environment (which sets also the initial state)
    env.reset()
    
    # sample state and action
    state = deepcopy(env.state)
    action = env.sample_action()
    
    # apply the first action
    prev_state = deepcopy(state)
    state, rew, _, info = env.step(action=deepcopy(action))
    end = bool(np.random.uniform(0,1) >= gamma)
    # execute the policy w.p. "gamma"
    while not end:
        prev_state = deepcopy(state)
        # simulation
        feat = dp.transform(state=deepcopy(state))
        action = pol.draw_action(state=deepcopy(feat))
        state, rew, _, info = env.step(action=deepcopy(action))
        # update 
        end = bool(np.random.uniform(0,1) >= gamma)
    # s_h = state
    s_h = prev_state
    a_h = action
    
    ### Qr AND Qg ###
    # execute the policy w.p. \sqrt{gamma}
    env.reset(state=deepcopy(s_h))
    state, rew, _, info = env.step(action=deepcopy(a_h))
    perf_r = rew
    perf_g = - info["costs"][0]
    end = bool(np.random.uniform(0,1) >= gamma)
    while not end:
        # simulation
        feat = dp.transform(state=deepcopy(state))
        action = pol.draw_action(state=deepcopy(feat))
        state, rew, _, info = env.step(action=deepcopy(action))
        cost = - info["costs"][0]
        # save
        perf_r += rew
        perf_g += cost
        # update
        end = bool(np.random.uniform(0,1) >= gamma)
        
    # compute Qr and Qg
    Qr = perf_r
    Qg = perf_g
    
    ### Vr and Vg ###
    Vr, Vg = value_estimation(
        env=deepcopy(env), pol=deepcopy(pol), dp=deepcopy(dp), 
        state=deepcopy(s_h)
    )
    
    ### Ar AND Ag ###
    Ar = Qr - Vr
    Ag = Qg - Vg
    
    return [s_h, a_h, Ar, Ag]


def value_estimation(env: BaseEnv, pol: BasePolicy, dp: BaseProcessor, state):
    # structures
        perf_r = 0
        perf_g = 0
        gamma = env.gamma
        end = bool(np.random.uniform(0,1) >= gamma)
        
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
            end = bool(np.random.uniform(0,1) >= np.sqrt(gamma))
        return [perf_r, perf_g]


# Base PG-PD
class BasePG_PD(ABC):
    def __init__(
        self,
        ite: int = 100,
        batch: int = 100,
        pol: TabularSoftmax = None,
        env: BaseEnv = None,
        lr: np.ndarray = None,
        lr_strategy: str = "constant",
        dp: BaseProcessor = None,
        threshold: float = 0,
        checkpoint_freq: int = 1000,
        n_jobs: int = 1,
        directory: str = None
    ) -> None:
        # check simulation parameters
        assert ite > 0
        self.ite = ite
        self.threshold = threshold
        self.checkpoint_freq = checkpoint_freq
        self.n_jobs = n_jobs
        
        check_directory_and_create(dir_name=directory)
        self.directory = directory
        
        assert len(lr) == 2
        self.lr_theta = lr[LearnRates.PARAM]
        self.lr_lambda = lr[LearnRates.LAMBDA]
        
        assert lr_strategy in ["constant", "adam"]
        self.lr_strategy = lr_strategy
        if self.lr_strategy == "adam":
            self.theta_adam = Adam(step_size=self.lr_theta, strategy="ascent")
            self.lambda_adam = Adam(step_size=self.lr_lambda, strategy="ascent")
        
        assert batch > 0
        self.batch = batch
        
        assert env is not None
        assert env.with_costs
        assert env.how_many_costs == 1
        # assert not env.continuous_env
        self.env = deepcopy(env)
        
        assert pol is not None
        self.pol = deepcopy(pol)
        
        assert dp is not None
        self.dp = deepcopy(dp)
    
    @abstractmethod
    def learn(self):
        pass
    
    @abstractmethod
    def save_results(self):
        pass
    


# NPG-PD
class NaturalPG_PD(BasePG_PD):
    # Note that it is thought for the Tabular Softmax
    # Note that this algorithm has the constraints as Vg >= b (with g in [0,1])
    # What we can do is to conver the cost and the threshold to be negative
    def __init__(
        self,
        ite: int = 100,
        batch: int = 100,
        pol: TabularSoftmax = None,
        env: BaseEnv = None,
        lr: np.ndarray = None,
        lr_strategy: str = "constant",
        dp: BaseProcessor = None,
        threshold: float = 0,
        checkpoint_freq: int = 1000,
        n_jobs: int = 1,
        directory: str = None,
        reg: float = 0
    ) -> None:
        super().__init__(
            ite=ite, env=env, pol=pol, dp=dp, batch=batch, lr=lr, 
            lr_strategy=lr_strategy, threshold=threshold, 
            checkpoint_freq=checkpoint_freq, n_jobs=n_jobs, directory=directory
        )
        self.obj_name = "npgpd"
        assert reg >= 0
        self.reg = reg
        if self.reg > 0:
            assert self.lr_lambda == self.lr_theta
            self.lr_strategy = "constant"
            self.obj_name = "rpgpd"
        
        # structures
        self.theta = np.zeros(self.env.state_dim * self.env.action_dim)
        self.lam = 0
        self.thetas = np.zeros(
            (self.ite, self.env.state_dim, self.env.action_dim), 
            dtype=np.float64
        )
        self.lambdas = np.zeros(self.ite, dtype=np.float64)
        self.values = np.zeros((self.ite, self.env.state_dim), dtype=np.float64)
        self.action_values = np.zeros(
            (self.ite, self.env.state_dim, self.env.action_dim),
            dtype=np.float64
        )
        self.adv_values = deepcopy(self.action_values)
        self.cost_values = np.zeros(self.ite, dtype=np.float64)
        self.perf = np.zeros(self.ite, dtype=np.float64)
        
        # additional parameters
        self.gamma = self.env.gamma
        assert self.gamma < 1, f"[{self.obj_name}] gamma must be less than one."
        self.eff_h = 1 / (1 - self.gamma)
        
        # cast the problem to cost minimization
        self.threshold = - self.threshold
        self.inner_batch = 1
        
    def learn(self):
        for t in tqdm(range(self.ite)):
            # save the theta history and lambda history
            self.thetas[t, :, :] = deepcopy(self.theta.reshape((self.env.state_dim, self.env.action_dim)))
            self.lambdas[t] = self.lam
            
            # set the policy parameters
            self.pol.set_parameters(thetas=deepcopy(self.theta), state=None, action=None)
            sampler_dict = dict(
                env=deepcopy(self.env),
                pol=deepcopy(self.pol),
                dp=deepcopy(self.dp),
                params=None,
                starting_state=None,
                starting_action=None,
                pol_values=bool(self.reg > 0)
            )
            
            # estimate V and Q values: structures
            q_estimation = np.zeros(len(self.env.discrete_state_space) *len(self.env.discrete_action_space), dtype=np.float64)
            v_estimation_dicts = []
            q_estimation_dicts = []
            for s_i, state_idx in enumerate(self.env.discrete_state_space):
                v_estimation_dicts.append(deepcopy(sampler_dict))
                v_estimation_dicts[s_i]["starting_state"] = deepcopy(state_idx)
                for _, action_idx in enumerate(self.env.discrete_action_space):
                    tmp_dict = deepcopy(v_estimation_dicts[s_i])
                    tmp_dict["starting_action"] = deepcopy(action_idx)
                    q_estimation_dicts.append(deepcopy(tmp_dict))
            # estimate V
            delayed_functions = delayed(pg_sampling_worker)
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**(v_estimation_dicts[ss_i])) for ss_i in range(len(self.env.discrete_state_space))
            )
            for ss_i in range(len(self.env.discrete_state_space)):
                perf = res[ss_i][TrajectoryResults.PERF] 
                cost = - res[ss_i][TrajectoryResults.CostInfo]["cost_perf"][0]
                pol_v = 0
                if self.reg > 0:
                    pol_v = res[i][TrajectoryResults.CostInfo]["pol_values"]
                self.values[t, ss_i] = perf + self.lam * cost - self.reg * pol_v
            # estimate Q
            delayed_functions = delayed(pg_sampling_worker)
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**(q_estimation_dicts[sa_i])) for sa_i in range(len(self.env.discrete_state_space) * len(self.env.discrete_action_space))
            )
            for sa_i in range(len(self.env.discrete_state_space) * len(self.env.discrete_action_space)):
                perf = res[sa_i][TrajectoryResults.PERF]
                cost = - res[sa_i][TrajectoryResults.CostInfo]["cost_perf"][0]
                pol_v = 0
                if self.reg > 0:
                    pol_v = res[i][TrajectoryResults.CostInfo]["pol_values"]
                q_estimation[sa_i] = perf + self.lam * cost - self.reg * pol_v
            self.action_values[t, :] = deepcopy(q_estimation.reshape((len(self.env.discrete_state_space), len(self.env.discrete_action_space))))
            
            # compute the advantage
            self.adv_values[t, :, :] = self.action_values[t, :, :] - self.values[t, :, np.newaxis]
            
            # estimate the sample cost
            sampler_dict["starting_state"] = None
            sampler_dict["starting_action"] = None
            perfs = np.zeros(self.batch, dtype=np.float64)
            costs = np.zeros(self.batch, dtype=np.float64)
            
            # parallel computation
            delayed_functions = delayed(pg_sampling_worker)
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**sampler_dict) for _ in range(self.batch)
            )
            for i in range(self.batch):
                perfs[i] = res[i][TrajectoryResults.PERF]
                costs[i] = - res[i][TrajectoryResults.CostInfo]["cost_perf"][0]
            self.perf[t] = np.mean(perfs)
            self.cost_values[t] = np.mean(costs)
            
            # update the parameters
            if self.lr_strategy == "constant":
                # self.theta = self.theta + (self.lr_theta * self.eff_h * self.adv_values[t,:,:]).flatten()
                self.theta = self.theta + (self.lr_theta * self.adv_values[t,:,:]).flatten()
                self.lam = np.clip((1 - self.reg * self.lr_lambda) * self.lam - self.lr_lambda * (self.cost_values[t] - self.threshold), 0, np.inf)
            elif self.lt_strategy == "adam":
                self.theta = self.theta + self.theta_adam.compute_gradient(self.eff_h * self.adv_values[t,:,:].flatten())
                self.lam = np.clip(self.lam - self.lambda_adam.compute_gradient(self.cost_values[t] - self.threshold), 0, np.inf)
            
            # save the results if the checkpoint has been reached
            if not (t % self.checkpoint_freq):
                self.save_results()
                
            print(f"[{self.obj_name}] values: {np.mean(self.values[t,:])}")
            print(f"[{self.obj_name}] perf: {self.perf[t]}")
            print(f"[{self.obj_name}] costs: {self.cost_values[t]}")
    
    def save_results(self):
        """Save the results."""
        results = {
            "perf": np.array(self.perf, dtype=float).tolist(),
            "v": np.array(self.values, dtype=float).tolist(),
            "q": np.array(self.action_values, dtype=float).tolist(),
            "adv": np.array(self.adv_values, dtype=float).tolist(),
            "costs": np.array(self.cost_values, dtype=float).tolist(),
            "theta_history": np.array(self.thetas, dtype=float).tolist(),
            "lambda_history": np.array(self.lambdas, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + f"/{self.obj_name}_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return
    

# RPG-PD
class NaturalPG_PD_2(BasePG_PD):
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
        self.obj_name = "npgpd2" if self.reg == 0 else "rpgpd2"
        
        # structures: running
        # policy parameters
        self.theta = np.zeros(self.pol.tot_params, dtype=np.float64)
        # lag multipliers
        self.lam = 0
        # q table
        self.omega = np.zeros(self.pol.tot_params, dtype=np.float64)
        self.omega_batch = np.zeros((self.batch, self.pol.tot_params), dtype=np.float64)
        
        # structures: history
        self.thetas = np.zeros(
            (self.ite, self.pol.tot_params), 
            dtype=np.float64
        )
        self.lambdas = np.zeros(self.ite, dtype=np.float64)
        self.cost_values = np.zeros(self.ite, dtype=np.float64)
        self.omegas = np.zeros((self.ite, self.env.state_dim * self.env.action_dim), dtype=np.float64)
        self.values = np.zeros(self.ite, dtype=np.float64)
        
        # additional parameters
        self.gamma = self.env.gamma
        assert self.gamma < 1, f"[{self.obj_name}] gamma must be less than one."
        
        # cast the problem to cost minimization
        self.threshold = - self.threshold
        assert inner_batch > 0
        self.inner_batch = inner_batch
        
    
    def learn(self):
        for t in tqdm(range(self.ite)):
            # save the theta history and lambda history
            self.thetas[t, :] = deepcopy(self.theta)
            self.lambdas[t] = self.lam
            
            # set the policy parameters
            self.pol.set_parameters(thetas=deepcopy(self.theta))
            sampler_dict = dict(
                env=deepcopy(self.env),
                pol=deepcopy(self.pol),
                dp=deepcopy(self.dp),
                params=None,
                starting_state=None,
                starting_action=None
            )
            
            # init to zero the omega values
            self.omega_r = np.zeros(self.pol.tot_params, dtype=np.float64)
            self.omega_r_batch = np.zeros((self.inner_batch, self.pol.tot_params), dtype=np.float64)
            self.omega_g = np.zeros(self.pol.tot_params, dtype=np.float64)
            self.omega_g_batch = np.zeros((self.inner_batch, self.pol.tot_params), dtype=np.float64)
            
            # unbiased estimaiton of Ar and Ag
            adv_dict = dict(
                env=deepcopy(self.env),
                pol=deepcopy(self.pol),
                dp=deepcopy(self.dp)
            )
            delayed_functions = delayed(advantage_estimation)
            adv_res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**adv_dict) for _ in range(self.inner_batch)
            )            
            alpha = 0
            for k in range(self.inner_batch):
                # sample q value
                # s, a, Ar, Ag = self._A_unbiased_estimation()
                s = adv_res[k][ADV_RES.s]
                a = adv_res[k][ADV_RES.a]
                Ar = adv_res[k][ADV_RES.Ar]
                Ag = adv_res[k][ADV_RES.Ag]
                
                # set the lr
                alpha = 2 / (self.inner_loop_param * (k + 1))
                
                # grad
                policy_score = self.pol.compute_score(state=deepcopy(s), action=deepcopy(a))
                # self.omega_r = self.omega_r - 2 * alpha * (self.omega_r @ policy_score - Ar) * policy_score
                # self.omega_g = self.omega_g - 2 * alpha * (self.omega_g @ policy_score - Ag) * policy_score
                self.omega_r = self.omega_r - 2 * alpha * (self.omega_r.T * policy_score - Ar) * policy_score
                self.omega_g = self.omega_g - 2 * alpha * (self.omega_g.T * policy_score - Ag) * policy_score
                
                # save vectors
                self.omega_r_batch[k, :] = deepcopy(self.omega_r)
                self.omega_g_batch[k, :] = deepcopy(self.omega_g)
            
            # unbiased estimation of Vr and Vg
            rews = np.zeros(self.batch, dtype=np.float64)
            costs = np.zeros(self.batch, dtype=np.float64)
            # delayed_functions = delayed(pg_sampling_worker)
            # res = Parallel(n_jobs=self.n_jobs, backend="loky")(
            #     delayed_functions(**sampler_dict) for _ in range(self.batch)
            # )
            # for i in range(self.batch):
            #     rews[i] = res[i][TrajectoryResults.PERF]
            #     costs[i] = - res[i][TrajectoryResults.CostInfo]["cost_perf"][0]
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
            
            # _, Vg = self._V_unbiased_estimation()
            # self.cost_values[t] = np.mean(Vg)
            # self.values[t] = np.mean(Vr)
            
            # compute the final omega for the iteration
            k_series = (np.arange(self.inner_batch, dtype=np.float64) + 1)[:, np.newaxis]
            omega_hat_r = (2 / (self.inner_batch * (self.inner_batch + 1))) * np.sum(k_series * self.omega_r_batch, axis=0)
            omega_hat_g = (2 / (self.inner_batch * (self.inner_batch + 1))) * np.sum(k_series * self.omega_g_batch, axis=0)
            omega_hat = omega_hat_r + self.lam * omega_hat_g
            self.omegas[t, :] = deepcopy(omega_hat)
            
            # update parameters
            if self.lr_strategy == "constant":    
                self.theta = self.theta + self.lr_theta * omega_hat
                self.lam = np.clip((1 - self.reg * self.lr_lambda) * self.lam - self.lr_lambda * (self.cost_values[t] - self.threshold), 0, np.inf)
            else:
                self.theta = self.theta + self.theta_adam.compute_gradient(omega_hat)
                self.lam = np.clip(self.lam - self.lambda_adam.compute_gradient(self.cost_values[t] - self.threshold + self.reg * self.lam), 0, np.inf)
            
            # save
            if not (t % self.checkpoint_freq):
                self.save_results()
                
            print(f"[{self.obj_name}] mean trajectory reward:\t {self.values[t]}")
            print(f"[{self.obj_name}] natural gradient values:\t {omega_hat}")
            print(f"[{self.obj_name}] mean trajectory cost:\t {self.cost_values[t]}")
            print(f"[{self.obj_name}] lambda:\t {self.lam}")
            print(f"[{self.obj_name}] theta:\t {self.theta}")
        
    def _A_unbiased_estimation(self):
        # structures
        perf_r = 0
        perf_g = 0
        gamma = self.env.gamma
        end = False
        
        ### SPOT sh AND ah ###
        # reset the environment (which sets also the initial state)
        self.env.reset()
        
        # sample state and action
        state = deepcopy(self.env.state)
        action = self.env.sample_action()
        
        # apply the first action
        prev_state = deepcopy(state)
        state, rew, _, info = self.env.step(action=deepcopy(action))
        end = bool(np.random.uniform(0,1) >= gamma)
        # execute the policy w.p. "gamma"
        while not end:
            prev_state = deepcopy(state)
            # simulation
            feat = self.dp.transform(state=deepcopy(state))
            action = self.pol.draw_action(state=deepcopy(feat))
            state, rew, _, info = self.env.step(action=deepcopy(action))
            # update 
            end = bool(np.random.uniform(0,1) >= gamma)
        # s_h = state
        s_h = prev_state
        a_h = action
        
        ### Qr AND Qg ###
        # execute the policy w.p. \sqrt{gamma}
        self.env.reset(state=deepcopy(s_h))
        state, rew, _, info = self.env.step(action=deepcopy(a_h))
        perf_r = rew
        perf_g = - info["costs"][0]
        end = bool(np.random.uniform(0,1) >= gamma)
        while not end:
            # simulation
            feat = self.dp.transform(state=deepcopy(state))
            action = self.pol.draw_action(state=deepcopy(feat))
            state, rew, _, info = self.env.step(action=deepcopy(action))
            cost = - info["costs"][0]
            # save
            perf_r += rew
            perf_g += cost
            # update
            end = bool(np.random.uniform(0,1) >= gamma)
            
        # compute Qr and Qg
        Qr = perf_r
        Qg = perf_g
        
        ### Vr and Vg ###
        Vr, Vg = self._V_unbiased_estimation(state=deepcopy(s_h))
        
        ### Ar AND Ag ###
        Ar = Qr - Vr
        Ag = Qg - Vg
        
        return s_h, a_h, Ar, Ag
    
    def _V_unbiased_estimation(self, state=None):
        # structures
        perf_r = 0
        perf_g = 0
        gamma = self.env.gamma
        end = bool(np.random.uniform(0,1) >= gamma)
        
        # reset the environment (which sets also the initial state)
        if state is None:
            self.env.reset()
        else:
            self.env.reset(state=deepcopy(state))
        
        # sample state and action
        state = deepcopy(self.env.state)
        while not end:
            feat = self.dp.transform(state=deepcopy(state))
            # simulation
            action = self.pol.draw_action(state=deepcopy(feat))
            state, rew, _, info = self.env.step(action=deepcopy(action))
            cost = - info["costs"][0]
            # save
            perf_r += rew
            perf_g += cost
            # update
            end = bool(np.random.uniform(0,1) >= np.sqrt(gamma))
        return perf_r, perf_g
    
    def save_results(self):
        """Save the results."""
        results = {
            "v": np.array(self.values, dtype=float).tolist(),
            "adv": np.array(self.omegas, dtype=float).tolist(),
            "costs": np.array(self.cost_values, dtype=float).tolist(),
            "theta_history": np.array(self.thetas, dtype=float).tolist(),
            "lambda_history": np.array(self.lambdas, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + f"/{self.obj_name}_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return