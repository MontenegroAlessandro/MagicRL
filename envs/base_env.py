"""todo"""
from abc import ABC, abstractmethod

class BaseEnv(ABC):
    """Abstract class for the Environments"""

    def __init__(self, horizon: int = 0, gamma: float = 0.99, verbose: bool = False) -> None:
        """todo"""
        self.horizon = horizon
        assert 0 <= gamma <= 1, "[ERROR] Invalid Discount Factor value."
        self.gamma = gamma
        self.time = 0
        self.verbose = verbose

    @abstractmethod
    def step(self):
        """todo"""
        pass

    def reset(self) -> None:
        """todo"""
        self.time = 0

    @abstractmethod
    def sample_random_state(self, n_samples):
        pass
