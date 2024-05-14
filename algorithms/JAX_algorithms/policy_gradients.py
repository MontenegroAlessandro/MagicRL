"""
Interface for Policy Gradient Algorithms implemented in JAX.
"""

# Libraries
from abc import ABC, abstractmethod
from envs.base_env import BaseEnv
from policies.JAX_policies.base_policy_jax import BasePolicyJAX
from data_processors import BaseProcessor, IdentityDataProcessor
from algorithms.utils import check_directory_and_create, PolicyGradientAlgorithms
import jax.numpy as jnp


class PolicyGradients(ABC):
    def __init__(self,
                 alg: str = PolicyGradientAlgorithms.PG,
                 lr: jnp.array = None,
                 ite: int = None,
                 batch_size: int = 1,
                 env: BaseEnv = None,
                 policy: BasePolicyJAX = None,
                 data_processor: BaseProcessor = IdentityDataProcessor(),
                 natural: bool = False,
                 lr_strategy: str = "constant",
                 checkpoint_freq: int = 1,
                 verbose: bool = False,
                 sample_deterministic_curve: bool = False,
                 directory: str = None) -> None:
        """
          Summary:
          Initialization of the Policy Gradients class.
          Args:
          alg (str): the algorithm. Default is "[PG]".

          lr (float): the learning rate. Default is None.

          ite (int): the number of iterations. Default is None.

          batch_size (int): the size of the batch. Default is 1.

          env (BaseEnv): the environment. Default is None.

          policy (BasePolicy): the policy. Default is None.

          data_processor (BaseProcessor): the data processor. Default is IdentityDataProcessor().

          natural (bool): whether to use natural gradients. Default is False.

          lr_strategy (str): the learning rate strategy. Default is "constant".

          checkpoint_freq (int): the frequency of checkpoints. Default is 1.

          verbose (bool): whether to print information. Default is False.

          sample_deterministic_curve (bool): whether to sample a deterministic curve. Default is False.

          directory (str): the directory to save the results. Default is None.
          """

        err_msg = "[ERROR] alg must be among [PG], [PGPE], [CPG], [CPGPE]"
        assert alg in ["PG", "PGPE", "CPG", "CPGPE"], err_msg
        self.alg = alg

        err_msg = "[" + alg + "]" + " lr must be positive!"
        assert lr[0] > 0, err_msg
        self.lr = lr[0]

        err_msg =  "[" + alg + "]"  +  " lr_strategy not valid!"
        assert lr_strategy in ["constant", "adam"], err_msg
        self.lr_strategy = lr_strategy

        err_msg = "[" + alg + "]" + " ite must be positive!"
        assert ite > 0, err_msg
        self.ite = ite

        err_msg = "[" + alg + "]" + " batch_size must be positive!"
        assert batch_size > 0, err_msg
        self.batch_size = batch_size

        err_msg = "[" + alg + "]" + " env is None"
        assert env is not None, err_msg
        self.env = env

        err_msg = "[" + alg + "]" + " policy is None"
        assert policy is not None, err_msg
        self.policy = policy

        err_msg = "[" + alg + "]" + " data_processor is None"
        assert data_processor is not None, err_msg
        self.data_processor = data_processor

        err_msg = "[" + alg + "]" + " checkpoint_freq must be positive!"
        assert checkpoint_freq > 0, err_msg
        self.checkpoint_freq = checkpoint_freq

        check_directory_and_create(dir_name=directory)
        self.directory = directory

        self.natural = natural
        self.verbose = verbose
        self.sample_deterministic_curve = sample_deterministic_curve

        # Useful structures
        self.time = 0
        self.performance_idx = jnp.zeros(ite, dtype=jnp.float64)

        return

    def _objective_function(self, **params) -> None:
        """
          Summary:
            This function computes the objective function.
          """
        pass

    def _sample_deterministic_curve(self):
        """
          Summary:
               This sample computes the deterministic curve associated with the
               sequence of parameters/hyperparameters configuration seen during the learning.
          """
        pass

    def learn(self):
        """Learning Function."""
        pass

    def save_results(self) -> None:
        """
          Summary:
               This function saves the results in a json file.
          """
        pass
