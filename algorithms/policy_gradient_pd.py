"""
Implmentation of NPG-PD and RPG-PD.
These algorithms only work for DISCRETE state and action spaces.
References: Ding et al. (sample-based versions with tabular softmax).
"""
# Libraries
import numpy as np
from joblib import Parallel, delayed
from data_processors.base_processor import BaseProcessor
from policies.softmax_policy import TabularSoftmax
from envs.base_env import BaseEnv
from abc import ABC, abstractmethod
from copy import deepcopy
from algorithms.samplers import pg_sampling_worker
from algorithms.utils import TrajectoryResults, LearnRates, check_directory_and_create
from tqdm import tqdm
from adam.adam import Adam
import io, json


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
        if self.lr_strategy:
            self.theta_adam = Adam(step_size=self.lr_theta, strategy="ascent")
            self.lambda_adam = Adam(step_size=self.lr_lambda, strategy="ascent")
        
        assert batch > 0
        self.batch = batch
        
        assert env is not None
        assert env.with_costs
        assert env.how_many_costs == 1
        assert not env.continuous_env
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
            
            # estimate V and Q values
            for state_idx in self.env.discrete_state_space:
                sampler_dict["starting_state"] = deepcopy(state_idx)
                perf = np.zeros(self.inner_batch, dtype=np.float64)
                costs = np.zeros(self.inner_batch, dtype=np.float64)
                pol_values = np.zeros(self.inner_batch, dtype=np.float64)
                
                # parallel computation
                delayed_functions = delayed(pg_sampling_worker)
                res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                    delayed_functions(**sampler_dict) for _ in range(self.inner_batch)
                )
                for i in range(self.inner_batch):
                    perf[i] = res[i][TrajectoryResults.PERF]
                    costs[i] = - res[i][TrajectoryResults.CostInfo]["cost_perf"][0]
                    if self.reg > 0:
                        pol_values[i] = res[i][TrajectoryResults.CostInfo]["pol_values"]
                    
                # V(s) estimation
                self.values[t, state_idx] = np.mean(perf + self.lam * costs - self.reg * pol_values)
                
                # Q(s, *) estimation
                for action_idx in self.env.discrete_action_space:
                    sampler_dict["starting_action"] = deepcopy(action_idx)
                    perf = np.zeros(self.inner_batch, dtype=np.float64)
                    costs = np.zeros(self.inner_batch, dtype=np.float64)
                    pol_values = np.zeros(self.inner_batch, dtype=np.float64)
                    
                    # parallel computation
                    delayed_functions = delayed(pg_sampling_worker)
                    res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                        delayed_functions(**sampler_dict) for _ in range(self.inner_batch)
                    )
                    for i in range(self.inner_batch):
                        perf[i] = res[i][TrajectoryResults.PERF]
                        costs[i] = - res[i][TrajectoryResults.CostInfo]["cost_perf"][0]
                        if self.reg > 0:
                            pol_values[i] = res[i][TrajectoryResults.CostInfo]["pol_values"]
                        
                    self.action_values[t, state_idx, action_idx] = np.mean(perf + self.lam * costs - self.reg * pol_values)
                    
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
class RegularizedPG_PD(BasePG_PD):
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
        inner_loop_param: float = 1,
        reg: float = 0
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
        
        # structures: running
        # policy parameters
        self.theta = np.zeros(self.env.state_dim * self.env.action_dim)
        # lag multipliers
        self.lam = 0
        # q table
        self.omega = np.zeros(self.env.state_dim * self.env.action_dim, dtype=np.float64)
        self.omega_batch = np.zeros((self.batch, self.env.state_dim * self.env.action_dim), dtype=np.float64)
        
        # structures: history
        self.thetas = np.zeros(
            (self.ite, self.env.state_dim, self.env.action_dim), 
            dtype=np.float64
        )
        self.lambdas = np.zeros(self.ite, dtype=np.float64)
        self.cost_values = np.zeros(self.ite, dtype=np.float64)
        self.omegas = np.zeros((self.ite, self.env.state_dim * self.env.action_dim), dtype=np.float64)
        self.values = np.zeros(self.ite, dtype=np.float64)
        
        # additional parameters
        self.gamma = self.env.gamma
        assert self.gamma < 1, f"[RPGPD] gamma must be less than one."
        self.eff_h = 1 / (1 - self.gamma)
        
        # cast the problem to cost minimization
        self.threshold = - self.threshold
        
    
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
                starting_action=None
            )
            
            # init to zero the omega values
            self.omega = np.zeros(self.env.state_dim * self.env.action_dim, np.float64)
            self.omega_batch = np.zeros((self.batch, self.env.state_dim * self.env.action_dim), dtype=np.float64)
            
            # unbiased estimaiton of Q_l(s,a) (for some (s,a)) [cannot be parallelized]
            for k in range(self.batch):
                # sample q value
                s, a, q = self._q_unbiased_estimation()
                # s and a are expected to be indices
                
                # set the lr
                alpha = 2 / (self.inner_loop_param * (k + 2))
                
                # take (one-hot) features
                feat = np.zeros((self.env.state_dim, self.env.action_dim), dtype=np.float64)
                feat[s,a] = 1
                feat = feat.flatten()
                
                # grad
                # self.omega = np.clip(self.omega - 2 * alpha * (feat * self.omega - q) * feat, 0, np.inf)
                self.omega = self.omega - 2 * alpha * (feat * self.omega - q) * feat
                self.omega_batch[k, :] = deepcopy(self.omega)
            
            # unbiased estimation of Vr and Vg [can be parallelized]
            rews = np.zeros(self.batch, dtype=np.float64)
            costs = np.zeros(self.batch, dtype=np.float64)
            delayed_functions = delayed(pg_sampling_worker)
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed_functions(**sampler_dict) for _ in range(self.batch)
            )
            for i in range(self.batch):
                rews[i] = res[i][TrajectoryResults.PERF]
                costs[i] = - res[i][TrajectoryResults.CostInfo]["cost_perf"][0]
                # costs[i] = res[i][TrajectoryResults.CostInfo]["cost_perf"][0]
            self.cost_values[t] = np.mean(costs)
            self.values[t] = np.mean(rews)
            """res = pg_sampling_worker(**sampler_dict)
            self.cost_values[t] = - res[TrajectoryResults.CostInfo]["cost_perf"][0]
            self.values[t] = res[TrajectoryResults.PERF]"""
            
            # compute the final omega for the iteration
            k_series = (np.arange(self.batch, dtype=np.float64) + 1)[:, np.newaxis]
            omega_hat = (2 / (self.batch * (self.batch + 1))) * np.sum(k_series * self.omega_batch, axis=0)
            self.omegas[t, :] = omega_hat # just keep the last one
            # self.values[t] = np.mean(omega_hat)
            
            # update parameters
            self.theta = self.lr_theta * np.sum(self.omegas, axis=0) 
            self.lam = np.clip((1 - self.lr_lambda * self.reg) * self.lam - self.reg * (self.cost_values[t] - self.threshold), 0, np.inf)
            
            # save
            if not (t % self.checkpoint_freq):
                self.save_results()
                
            print(f"[RPGPD] mean trajectory reward:\t {self.values[t]}")
            print(f"[RPGPD] mean lagrangian values:\t {np.mean(omega_hat)}")
            print(f"[RPGPD] mean trajectory cost:\t {self.cost_values[t]}")
            print(f"[RPGPD] lambda:\t {self.lam}")
            print(f"[RPGPD] lag values:\t {omega_hat}")
        
    def _q_unbiased_estimation(self):
        # structures
        rew = 0
        cost = 0
        q_value = 0
        gamma = self.env.gamma
        end = False
        
        # reset the environment (which sets also the initial state)
        self.env.reset()
        
        # sample state and action
        state = self.dp.transform(state=self.env.state)
        action = self.env.sample_action()
        
        # apply the first action
        state, rew, _, info = self.env.step(action=action)
        state = self.dp.transform(state=self.env.state)
        end = bool(np.random.uniform(0,1) >= gamma)
        
        # execute the policy w.p. "gamma"
        h = 1
        while not end:
            # simulation
            action = self.pol.draw_action(state=state)
            state, rew, _, info = self.env.step(action=action)
            state = self.dp.transform(state=self.env.state)
            # update 
            end = bool(np.random.uniform(0,1) >= gamma)
            h += 1
        s_h = state
        a_h = action
        k = h+1
        
        # execute the policy w.p. \sqrt{gamma}
        end = bool(np.random.uniform(0,1) >= np.sqrt(gamma))
        while not end:
            # simulation
            action = self.pol.draw_action(state=state)
            state, rew, _, info = self.env.step(action=action)
            state = self.dp.transform(state=self.env.state)
            cost = - info["costs"][0]
            # cost = info["costs"][0]
            # save
            q_value += (gamma ** (0.5 * (k-h))) * (rew + self.lam * (cost - self.threshold) - self.reg * np.log(self.pol.compute_pi(state=state, action=action)))
            # update
            end = bool(np.random.uniform(0,1) >= np.sqrt(gamma))
            k += 1
            
        # compute q_value
        q_value = q_value - self.reg * np.log(self.pol.compute_pi(state=s_h, action=a_h))
        
        return s_h, a_h, q_value
    
    def save_results(self):
        """Save the results."""
        results = {
            "v": np.array(self.values, dtype=float).tolist(),
            "q": np.array(self.omegas, dtype=float).tolist(),
            "costs": np.array(self.cost_values, dtype=float).tolist(),
            "theta_history": np.array(self.thetas, dtype=float).tolist(),
            "lambda_history": np.array(self.lambdas, dtype=float).tolist()
        }

        # Save the json
        name = self.directory + "/rpgpd_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()
        return