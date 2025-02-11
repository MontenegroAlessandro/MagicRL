import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tempfile
import shutil
import argparse
import copy
import logging
from pathlib import Path
from abc import ABC, abstractmethod
import time
import uuid
import filelock

# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from algorithms.off_policy import OffPolicyGradient
from algorithms import PolicyGradient
from envs import Swimmer, HalfCheetah, Reacher, Humanoid, Ant, Hopper, Pendulum
from policies import LinearGaussianPolicy
from data_processors import IdentityDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnvType(Enum):
    SWIMMER = 'swimmer'
    HALF_CHEETAH = 'half_cheetah'
    REACHER = 'reacher'
    HUMANOID = 'humanoid'
    ANT = 'ant'
    HOPPER = 'hopper'
    PENDULUM = 'pendulum'

@dataclass(frozen=True)
class AlgorithmConfig:
    """Configuration for a single algorithm."""
    name: str
    batch_sizes: Tuple[int, ...]
    window_lengths: Optional[Tuple[int, ...]] = None  # Only used for off_pg

@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable configuration for experiment parameters."""
    algorithm_1: AlgorithmConfig
    algorithm_2: AlgorithmConfig
    n_trials: int
    horizon: int
    ite: int
    checkpoint_dir: Path
    n_workers: int
    env_name: EnvType
    var: float
    learning_rate: float = 0.003
    run_id: str = ""

    def __post_init__(self):
        """Validate configuration parameters and set run_id if not provided."""
        if self.n_trials <= 0:
            raise ValueError("Number of trials must be positive")
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        
        object.__setattr__(self, 'run_id', 
            self.run_id or f"{int(time.time())}_{uuid.uuid4().hex[:6]}")

class Environment(ABC):
    """Abstract base class for environments."""
    @abstractmethod
    def create(self, horizon: int) -> Any:
        pass

class GymEnvironment(Environment):
    """Factory for creating Gym-based environments."""
    def __init__(self, env_type: EnvType):
        self.env_type = env_type
        self._env_mapping = {
            EnvType.SWIMMER: Swimmer,
            EnvType.HALF_CHEETAH: HalfCheetah,
            EnvType.REACHER: Reacher,
            EnvType.HUMANOID: Humanoid,
            EnvType.ANT: Ant,
            EnvType.HOPPER: Hopper,
            EnvType.PENDULUM: Pendulum
        }

    def create(self, horizon: int) -> Any:
        env_class = self._env_mapping.get(self.env_type)
        if not env_class:
            raise ValueError(f"Unsupported environment type: {self.env_type}")
        return env_class(horizon=horizon, gamma=1, render=False, clip=True)

class Algorithm(ABC):
    """Abstract base class for RL algorithms."""
    @abstractmethod
    def create(self, config: ExperimentConfig, env: Any, 
               policy: LinearGaussianPolicy, directory: Path, **kwargs) -> Any:
        pass

class PolicyGradientAlgorithm(Algorithm):
    def create(self, config: ExperimentConfig, env: Any, 
               policy: LinearGaussianPolicy, directory: Path, **kwargs) -> Any:
        tot_params = env.state_dim * env.action_dim
        return PolicyGradient(
            lr=[config.learning_rate],
            lr_strategy="adam",
            estimator_type="GPOMDP",
            initial_theta=np.zeros(tot_params),
            ite=config.ite,
            batch_size=kwargs.get('batch_size'),
            env=copy.deepcopy(env),
            policy=copy.deepcopy(policy),
            data_processor=IdentityDataProcessor(),
            directory=str(directory),
            verbose=False,
            natural=False,
            checkpoint_freq=100,
            n_jobs=config.n_workers,
        )

class OffPolicyGradientAlgorithm(Algorithm):
    def create(self, config: ExperimentConfig, env: Any, 
               policy: LinearGaussianPolicy, directory: Path, **kwargs) -> Any:
        tot_params = env.state_dim * env.action_dim
        return OffPolicyGradient(
            lr=[config.learning_rate],
            lr_strategy="adam",
            initial_theta=np.zeros(tot_params),
            ite=config.ite,
            batch_size=kwargs.get('batch_size'),
            env=copy.deepcopy(env),
            policy=copy.deepcopy(policy),
            data_processor=IdentityDataProcessor(),
            directory=str(directory),
            verbose=False,
            natural=False,
            checkpoint_freq=100,
            n_jobs=config.n_workers,
            window_length=kwargs.get('window_length', 1),
        )

class PolicyFactory:
    """Factory for creating policies."""
    @staticmethod
    def create(s_dim: int, a_dim: int, var: float) -> LinearGaussianPolicy:
        tot_params = s_dim * a_dim
        return LinearGaussianPolicy(
            parameters=np.zeros(tot_params),
            dim_state=s_dim,
            dim_action=a_dim,
            std_dev=np.sqrt(var),
            std_decay=0,
            std_min=1e-5,
            multi_linear=True,
        )

class CheckpointManager:
    """Manages experiment checkpoints."""
    def __init__(self, config: ExperimentConfig):
        # Include run_id in checkpoint file name
        checkpoint_filename = (
            f'{config.algorithm_1.name}_vs_{config.algorithm_2.name}'
            f'_run_{config.run_id}_checkpoint.pkl'
        )
        self.checkpoint_file = config.checkpoint_dir / checkpoint_filename
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add file locking
        self.lock_file = self.checkpoint_file.with_suffix('.lock')
        self._lock = filelock.FileLock(str(self.lock_file))

    def load(self) -> Tuple[Optional[Dict], int]:
        if not self.checkpoint_file.exists():
            return None, 0

        try:
            with self._lock:
                with open(self.checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    logger.info(f"Resuming from trial {checkpoint_data['last_trial'] + 1}")
                    return checkpoint_data['results'], checkpoint_data['last_trial'] + 1
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None, 0

    def save(self, results: Dict, last_trial: int):
        checkpoint_data = {'results': results, 'last_trial': last_trial}
        with self._lock:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)

    def cleanup(self):
        with self._lock:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            if self.lock_file.exists():
                self.lock_file.unlink()

class ExperimentRunner:
    """Manages the execution of experiments."""
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config)
        self.env_factory = GymEnvironment(config.env_name)
        self.algorithms = {
            'pg': PolicyGradientAlgorithm(),
            'off_pg': OffPolicyGradientAlgorithm()
        }

    def _get_algorithm_key(self, alg_config: AlgorithmConfig, 
                          batch_size: int, window_length: Optional[int] = None) -> str:
        """Generate a unique key for algorithm results."""
        if alg_config.name == 'off_pg' and window_length is not None:
            return f'off_pg_w{window_length}_b{batch_size}'
        return f'{alg_config.name}_b{batch_size}'

    def initialize_results(self) -> Dict:
        """Initialize results dictionary based on algorithm configurations."""
        results = {}
        
        # Initialize results for both algorithms
        for alg_config in [self.config.algorithm_1, self.config.algorithm_2]:
            if alg_config.name == 'off_pg':
                for w in alg_config.window_lengths:
                    for b in alg_config.batch_sizes:
                        results[self._get_algorithm_key(alg_config, b, w)] = []
            else:
                for b in alg_config.batch_sizes:
                    results[self._get_algorithm_key(alg_config, b)] = []
        
        return results

    def run_single_algorithm(self, alg_config: AlgorithmConfig, trial_dir: Path, 
                           env: Any, policy: LinearGaussianPolicy, 
                           results: Dict) -> Dict:
        """Run a single algorithm with all its configurations."""
        if alg_config.name == 'off_pg':
            for window_length in alg_config.window_lengths:
                for batch_size in alg_config.batch_sizes:
                    key = self._get_algorithm_key(alg_config, batch_size, window_length)
                    alg = self.algorithms[alg_config.name].create(
                        self.config, env, policy,
                        trial_dir / key,
                        window_length=window_length,
                        batch_size=batch_size
                    )
                    alg.learn()
                    results[key].append(alg.performance_idx)
        else:
            for batch_size in alg_config.batch_sizes:
                key = self._get_algorithm_key(alg_config, batch_size)
                alg = self.algorithms[alg_config.name].create(
                    self.config, env, policy,
                    trial_dir / key,
                    batch_size=batch_size
                )
                alg.learn()
                results[key].append(alg.performance_idx)
        
        return results

    def run_algorithms(self, trial_dir: Path, env: Any, 
                      policy: LinearGaussianPolicy, results: Dict) -> Dict:
        """Run both algorithms with their respective configurations."""
        results = self.run_single_algorithm(
            self.config.algorithm_1, trial_dir, env, policy, results
        )
        results = self.run_single_algorithm(
            self.config.algorithm_2, trial_dir, env, policy, results
        )
        return results

    def run(self) -> Dict:
        """Execute the experiment."""
        results, start_trial = self.checkpoint_manager.load()
        if results is None:
            results = self.initialize_results()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            env = self.env_factory.create(self.config.horizon)
            policy = PolicyFactory.create(
                env.state_dim, env.action_dim, self.config.var
            )

            for trial in range(start_trial, self.config.n_trials):
                logger.info(f"Running trial {trial + 1}/{self.config.n_trials}")
                trial_dir = temp_path / f"trial_{trial}"
                trial_dir.mkdir(parents=True, exist_ok=True)

                try:
                    results = self.run_algorithms(trial_dir, env, policy, results)
                    self.checkpoint_manager.save(results, trial)
                except Exception as e:
                    logger.error(f"Error in trial {trial}: {e}")
                    self.checkpoint_manager.save(results, trial - 1)
                    raise

        self.checkpoint_manager.cleanup()
        return results

class Visualizer:
    """Handles experiment visualization."""
    @staticmethod
    def plot_comparison(results: Dict, window: int = 10, 
                       save_path: str = 'comparison_plot.png'):
        plt.figure(figsize=(10, 6))
        
        # Get unique window lengths from results keys
        window_lengths = sorted([int(k.split('_')[-1]) for k in results.keys() if k != 'pg'])
        
        # Generate color map dynamically
        colors = {'pg': 'blue'}
        # Color map for off-policy variants
        color_map = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(window_lengths)))
        for i, w in enumerate(window_lengths):
            colors[f'off_pg_w{w}_b{window_lengths[i]}'] = color_map[i]
        
        # Sort keys to ensure consistent plotting order
        for alg_name in sorted(results.keys()):
            curves = np.array(results[alg_name])
            
            # Create readable label
            if alg_name == 'pg':
                label = 'PG'
            else:
                window_size = alg_name.split('_')[-1]
                label = f'Off-PG (w={window_size})'
            
            # Compute mean and std across trials
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            
            # Smooth curves
            if window > 1:
                kernel = np.ones(window) / window
                mean_curve = np.convolve(mean_curve, kernel, mode='valid')
                std_curve = np.convolve(std_curve, kernel, mode='valid')
                x = np.arange(len(mean_curve))
            else:
                x = np.arange(len(mean_curve))
                
            # Plot mean and confidence interval
            plt.plot(x, mean_curve, label=label, color=colors.get(alg_name, 'gray'))
            plt.fill_between(x, 
                            mean_curve - std_curve,
                            mean_curve + std_curve,
                            alpha=0.2,
                            color=colors.get(alg_name, 'gray'))
        
        plt.xlabel('Iteration')
        plt.ylabel('Performance')
        plt.title('Policy Gradient Methods Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def parse_args() -> ExperimentConfig:
    """Parse command line arguments and create ExperimentConfig."""
    parser = argparse.ArgumentParser(description='Compare RL algorithms')
    
    # Algorithm 1 configuration
    parser.add_argument('--alg1', type=str, default='pg', 
                      help='First algorithm to compare')
    parser.add_argument('--batch_alg_1', type=int, nargs='+', default=[100],
                      help='Batch sizes for first algorithm')
    parser.add_argument('--window_alg_1', type=int, nargs='+', default=None,
                      help='Window lengths for first algorithm (if off_pg)')
    
    # Algorithm 2 configuration
    parser.add_argument('--alg2', type=str, default='off_pg', 
                      help='Second algorithm to compare')
    parser.add_argument('--batch_alg_2', type=int, nargs='+', default=[100, 200, 400],
                      help='Batch sizes for second algorithm')
    parser.add_argument('--window_alg_2', type=int, nargs='+', default=[2, 4, 8],
                      help='Window lengths for second algorithm (if off_pg)')
    
    # Other parameters
    parser.add_argument('--n_trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--ite', type=int, default=100, help='Number of iterations')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to store checkpoints')
    parser.add_argument("--horizon", help="The horizon amount.", type=int,default=100)
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--env', type=str, default='swimmer',
                      choices=['swimmer', 'half_cheetah', 'reacher', 'humanoid', 'ant', 'hopper', 'pendulum'],
                      help='Environment to use')
    parser.add_argument('--var', type=float, default=1, help='The exploration amount (variance)')
    parser.add_argument('--lr', type=float, default=0.003, 
                      help='Learning rate for the algorithms')
    parser.add_argument('--run-id', type=str, default="",
                       help='Unique identifier for this run')
    
    args = parser.parse_args()
    
    # Create algorithm configurations with proper window lengths
    alg1_config = AlgorithmConfig(
        name=args.alg1,
        batch_sizes=tuple(args.batch_alg_1),
        window_lengths=tuple(args.window_alg_1 if args.window_alg_1 is not None else [1]) 
            if args.alg1 == 'off_pg' else None
    )
    
    alg2_config = AlgorithmConfig(
        name=args.alg2,
        batch_sizes=tuple(args.batch_alg_2),
        window_lengths=tuple(args.window_alg_2 if args.window_alg_2 is not None else [1])
            if args.alg2 == 'off_pg' else None
    )
    
    return ExperimentConfig(
        algorithm_1=alg1_config,
        algorithm_2=alg2_config,
        n_trials=args.n_trials,
        horizon=args.horizon,
        ite=args.ite,
        checkpoint_dir=Path(args.checkpoint_dir),
        n_workers=args.n_workers,
        env_name=EnvType(args.env),
        var=args.var,
        learning_rate=args.lr,
        run_id=args.run_id
    )

def main():
    """Main entry point."""
    runner = None
    try:
        config = parse_args()
        logger.info(f"Starting experiment with run ID: {config.run_id}")
        
        runner = ExperimentRunner(config)
        results = runner.run()
        
        var_str = str(config.var).replace(".", "")
        logger.info("Saving comparison plot...")
        
        save_path = (
            f'{config.env_name.value}_{config.algorithm_1.name}_vs_'
            f'{config.algorithm_2.name}_var_{var_str}_run_{config.run_id}.png'
        )
            
        Visualizer.plot_comparison(results, window=10, save_path=save_path)
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    finally:
        if runner and hasattr(runner, 'checkpoint_manager'):
            runner.checkpoint_manager.cleanup()

if __name__ == '__main__':
    main()
