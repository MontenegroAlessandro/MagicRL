from typing import List, Tuple, Callable
from numpy.random import default_rng
from torch import (
    eye, inverse, zeros, clamp, concat, cartesian_prod, tensor, ger, flatten, diag, einsum, Tensor
)
from torch.linalg import lstsq
import numpy as np
import torch
from typing import Tuple, Callable
from torch import zeros, Tensor

class RobotWorld:
    def __init__(self, range_pos, range_vel) -> None:
        self.s_r = torch.tensor(Config.s_r)
        self.rng = np.random.default_rng()
        self.range_pos = range_pos
        self.range_vel = range_vel
        self.A, self.B = self.generate_dynamics(Config.time_step)

    def generate_dynamics(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        A = torch.tensor(
            [
                [1., 0, dt, 0],
                [0, 1., 0, dt],
                [0, 0, 1., 0],
                [0, 0, 0, 1.],
            ]
        ).double()

        B = torch.tensor(
            [
                [dt**2 / 2, 0.0],
                [0.0, dt**2 / 2],
                [dt, 0.0],
                [0.0, dt],
            ]
        ).double()

        return A, B

    def reset(self, n_samples: int=1):
        s = np.stack([
            self.rng.uniform(self.range_pos[0], self.range_pos[1], n_samples),
            self.rng.uniform(self.range_pos[0], self.range_pos[1], n_samples),
            self.rng.uniform(self.range_vel[0], self.range_vel[1], n_samples),
            self.rng.uniform(self.range_vel[0], self.range_vel[1], n_samples),
        ])
        self.s = torch.from_numpy(s).double().T
        return self.s

    def generate_noise(self, size: int) -> Tensor:
        return torch.tensor(
            self.rng.normal(
            scale=np.array(
                [
                    Config.noise_pos,
                    Config.noise_pos,
                    Config.noise_vel,
                    Config.noise_vel
                ]
            ) * Config.time_step,
            size=size,
        )
        )

    def step(self, u: np.ndarray) -> np.ndarray:
        noise = self.generate_noise(self.s.shape)
        self.s_noiseless = self.s @ self.A.T + u @ self.B.T
        self.s = self.s_noiseless + noise
        return self.s

class Sampler:
    def __init__(self, env: RobotWorld, gamma: float) -> None:
        self.env = env
        self.gamma = gamma

    def sample_trajectory(self, K: Tensor, T: int) -> Tuple[Tensor, Tensor]:
        states = zeros([T, K.shape[1]])
        actions = zeros([T, K.shape[0]])

        s = self.env.reset()
        for i in range(0, T):
            u = s @ K.T
            sp = self.env.step(u)

            states[i] = s
            actions[i] = u

            s = sp
        return states, actions

    def rollout_V(self, s: Tensor, K: Tensor, n: int, reward_fn: Callable) -> float:
        self.env.reset(s.shape[0])
        self.env.s = s
        v = 0
        for i in range(n):
            a = s @ K.T
            v += (self.gamma ** i) * reward_fn(self.env, a)
            s = self.env.step(a)
        return v

    def rollout_Q(self, s: Tensor, a: Tensor, K: Tensor, n: int, reward_fn: Callable) -> float:
        self.env.reset(s.shape[0])
        self.env.s = s
        q = reward_fn(self.env, a)
        s = self.env.step(a)
        for i in range(1, n):
            a = s @ K.T
            q += (self.gamma**i) * reward_fn(self.env, a)
            s = self.env.step(a)
        return q.detach()

    def estimate_V_rho_closed(self, P: Tensor, n: int) -> float:
        s = self.env.reset(n)
        return ((s @ P) * s).sum(dim=1).mean().item()

    def estimate_V_rho_rollout(self, K: Tensor, n_samples: int, n_rollout: int, reward_fn: Callable) -> float:
        s = self.env.reset(n_samples)
        v = 0
        for i in range(n_rollout):
            a = s @ K.T
            v += (self.gamma ** i) * reward_fn(self.env, a)
            s = self.env.step(a)
        return v.mean().detach().item()



class Config:
    duration: float = 10.0
    time_step: float = 0.05
    x_range: Tuple[int, int] = [-10, 10]
    y_range: Tuple[int, int] = [-10, 10]
    vx_range: Tuple[int, int] = [-.1, .1]
    vy_range: Tuple[int, int] = [-.1, .1]
    noise_pos: float = 1.0
    noise_vel: float = 1.0
    noise_asset: float = 5.0
    s_r: List[int] = [0, 0, 0, 0]


class InventoryControl():
    def __init__(self, range_assets, range_demand, range_acq) -> None:
        self.rng = np.random.default_rng()
        self.range_assets = range_assets
        self.range_demand = range_demand
        self.range_acq = range_acq

    def reset(self, n_samples: int=1):
        self.demand = self.generate_noise([n_samples, 4])

        s = np.concatenate([
            self.rng.uniform(self.range_assets[0], self.range_assets[1], (n_samples, 4)),
            self.rng.uniform(self.range_demand[0], self.range_demand[1], (n_samples, 4)),
            self.rng.uniform(self.range_acq[0], self.range_acq[1], (n_samples, 4)),
        ], axis=1)
        self.s = torch.from_numpy(s).double()
        return self.s

    def generate_noise(self, size: int) -> Tensor:
        return torch.tensor(
            self.rng.normal(
                loc=10,
                scale=np.array(
                    [
                        Config.noise_asset,
                        Config.noise_asset,
                        Config.noise_asset,
                        Config.noise_asset,
                    ]
                ),
                size=size,
        )
        ).clip(min=0)

    def step(self, u: np.ndarray) -> np.ndarray:
        u = u.clip(min=0)

        self.s[:, :4] = torch.clip(self.s[:, :4] + u - self.demand, min=0.0)
        self.s[:, 4:8] = self.demand
        self.s[:, 8:] = u

        self.demand = self.generate_noise([self.s.shape[0], 4])

        return self.s

# This class implements the sample-based version of AD-PGPD
class ADpgpdSampled:
    def __init__(
            self,
            ds: int,
            da: int,
            env: RobotWorld,
            eta: float,
            tau: float,
            gamma: float,
            b: float,
            alpha: float,
            primal_reward_fn: Callable,
            primal_reward_reg_fn: Callable,
            dual_reward_fn: Callable,
            starting_pos_fn: Callable,
        ) -> None:
        self.env = env
        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.b = b
        self.alpha = alpha
        self.ds, self.da = ds, da

        self.primal_reward_fn = primal_reward_fn
        self.primal_reward_reg_fn = primal_reward_reg_fn
        self.dual_reward_fn = dual_reward_fn
        self.starting_pos_fn = starting_pos_fn

        self.sampler = Sampler(env, gamma)

    def policy_evaluation(self, K: Tensor, lmbda: Tensor, n_samples: int, n_rollout: int) -> Tensor:
        s, a = self.starting_pos_fn(n_samples)
        s_a = concat([s, a], dim=1)
        X = einsum("bi,bj->bij", s_a, s_a).view(n_samples, (self.ds + self.da)**2)

        def reward_fn(env, action):
            return self.primal_reward_fn(env, action) + self.primal_reward_reg_fn(env, action) + lmbda * self.dual_reward_fn(env, action)

        a_pi = s @ K.T
        q = self.sampler.rollout_Q(s, a, K, n_rollout, reward_fn=reward_fn)
        y = q + (1 / self.eta) * diag(a_pi @ a.T)

        theta = lstsq(X, y, driver='gelsd').solution
        return theta

    # This method implements the primal update in equation 9a using samples
    def primal_update(self, theta: Tensor) -> Tensor:
        W_1 = zeros((self.da, self.ds))
        for i in range(self.da):
            for j in range(self.ds):
                s_idx, a_idx = zeros(self.ds), zeros(self.da)
                s_idx[j] = -1
                a_idx[i] = 1
                s_a_idx = concat([s_idx, a_idx])
                mask = - cartesian_prod(s_a_idx, s_a_idx).prod(dim=1).clip(-1, 0)
                w = (theta * mask).sum()
                W_1[i, j] = w

        W_2 = zeros((self.da, self.da))
        for i in range(self.da):
            for j in range(self.da):
                s_idx, a_idx = zeros(self.ds), zeros(self.da)
                a_idx[i] = 1
                a_idx[j] = -1
                s_a_idx = concat([s_idx, a_idx])

                if i == j:
                    mask = cartesian_prod(s_a_idx, s_a_idx).prod(dim=1)
                    w = (theta * mask).sum()
                    W_2[i, j] = 2 * w * self.alpha
                else:
                    mask = - cartesian_prod(s_a_idx, s_a_idx).prod(dim=1).clip(-1, 0)
                    w = (theta * mask).sum()
                    W_2[i, j] = w
        K = - inverse(W_2 - (self.tau + 1 / self.eta) * eye(self.da)) @ W_1
        return K.double()

    # This method implements the dual update in equation 9b using samples
    def dual_update(self, K: Tensor, lmbda: Tensor, n_samples: int, n_rollout) -> Tensor:
        v = self.sampler.estimate_V_rho_rollout(K, n_samples, n_rollout, self.dual_reward_fn)
        return clamp(lmbda - self.eta * (v - self.b + self.tau * lmbda), min=0), v

    def train_unconstrained(self, epochs: int, n_pe: int, n_rho: int, n_roll: int) -> Tuple[Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        theta = zeros((self.ds + self.da)**2)
        K = self.primal_update(theta)
        for e in range(epochs):
            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.primal_reward_fn)
                loss_dual = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.dual_reward_fn)
                losses_primal.append(loss_primal)
                losses_dual.append(loss_dual)

            theta = self.policy_evaluation(K, 0, n_pe, n_roll)
            K = self.primal_update(theta)

            print(f"Episode {e}/{epochs} - Return {loss_primal} \r", end='')
        return K, losses_primal, losses_dual

    # This method iterates the primal and the dual updates
    def train_constrained(self, epochs: int, n_pe: int, n_rho: int, n_roll: int) -> Tuple[Tensor, Tensor, List[float], List[float]]:
        losses_primal, losses_dual = [], []
        theta = zeros((self.ds + self.da)**2)
        K = self.primal_update(theta)
        lmbda = zeros(1)
        for e in range(epochs):
            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.primal_reward_fn)
                losses_primal.append(loss_primal)

            theta = self.policy_evaluation(K, lmbda, n_pe, n_roll)
            K = self.primal_update(theta)
            lmbda, loss_dual = self.dual_update(K, lmbda, n_rho, n_roll)
            losses_dual.append(loss_dual)

            print(f'Episode {e}/{epochs} - Return {loss_primal} - Constrain {loss_dual} - Lambda {lmbda.detach().item()}\r', end='')
        return K, lmbda, losses_primal, losses_dual

    def resume_training(self, K, lmbda, losses_primal, losses_dual, epochs: int, n_pe: int, n_rho: int, n_roll: int):
        for e in range(epochs):
            if e % 10 == 0:
                loss_primal = self.sampler.estimate_V_rho_rollout(K, n_rho, n_roll, self.primal_reward_fn)
                losses_primal.append(loss_primal)

            theta = self.policy_evaluation(K, lmbda, n_pe, n_roll)
            K = self.primal_update(theta)
            lmbda, loss_dual = self.dual_update(K, lmbda, n_rho, n_roll)
            losses_dual.append(loss_dual)

            print(f'Episode {e}/{epochs} - Return {loss_primal} - Constrain {loss_dual} - Lambda {lmbda.detach().item()}\r', end='')
        return K, lmbda, losses_primal, losses_dual


def main():
    ds = 4
    da = 2

    b = - 1_000
    gamma = 0.99

    tau = 0.2

    G1 = - torch.tensor([1.0, 1.0, .001, .001]).double()
    G2 = - torch.tensor([.001, .001, 1.0, 1.0]).double()

    R1 = - torch.tensor([0.01, 0.01]).double()
    R2 = - torch.tensor([0.01, 0.01]).double()

    def primal_reward_fn(env, a):
        return (env.s.abs() * G1).sum(dim=1) + (a.abs() * R1).sum(dim=1)

    def primal_reward_reg_fn(env, a):
        return - (tau / 2) * (a * a).sum(dim=1)

    def dual_reward_fn(env, a):
        return ((env.s ** 2) * G2).sum(dim=1) + (tau / 2) + ((a ** 2) * R2).sum(dim=1)

    def starting_pos_fn(nsamples):
        rng = np.random.default_rng()

        s = torch.tensor(rng.uniform(
            low=[40, 40, -10, -10],
            high=[50, 50, 10, 10],
            size=[nsamples, 4],
        )).double()

        a = torch.tensor(rng.uniform(
            low=[-10, -10],
            high=[10, 10],
            size=[nsamples, 2],
        )).double()

        return s, a

    epochs = 40_000
    n_pe = 100
    n_rho = 1_000
    n_roll = 400

    alpha = 1.0
    eta = 0.00001

    env = RobotWorld(range_pos=[40, 50], range_vel=[-.1, .1])
    dpgpd = ADpgpdSampled(ds, da, env, eta, tau, gamma, b, alpha, primal_reward_fn, primal_reward_reg_fn,
                          dual_reward_fn, starting_pos_fn)

    K, lmbda, losses_primal, losses_dual = dpgpd.train_constrained(epochs, n_pe, n_rho, n_roll)

    np.save('../results/vel_sampled_primal.npy', losses_primal)
    np.save('../results/vel_sampled_dual.npy', losses_dual)


if __name__ == "__main__":
    main()