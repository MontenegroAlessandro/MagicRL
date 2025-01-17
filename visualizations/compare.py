import os
import sys
# Add the project root directory to Python path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import numpy as np
import matplotlib.pyplot as plt
from algorithms.off_policy import OffPolicyGradient
from algorithms import PolicyGradient
from envs import Swimmer
from policies import LinearGaussianPolicy
from data_processors import IdentityDataProcessor
import copy
import tempfile
import shutil
import argparse
import pickle

def run_experiment(alg1='pg', alg2='off_pg', n_trials=5, horizon=100, ite=100, window_lengths=[3, 5, 10], checkpoint_dir='checkpoints', n_workers=1):
    """
    Run algorithms multiple times and return their performance curves.
    Includes checkpoint system to resume from crashes.
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f'{alg1}_vs_{alg2}_checkpoint.pkl')
    
    # Try to load existing results from checkpoint
    results = None
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
                results = checkpoint_data['results']
                start_trial = checkpoint_data['last_trial'] + 1
                print(f"Resuming from trial {start_trial}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            start_trial = 0
    
    # Initialize results if no checkpoint was loaded
    if results is None:
        start_trial = 0
        results = {alg1: []}
        if alg2 == 'off_pg':
            for w in window_lengths:
                results[f'off_pg_{w}'] = []
        else:
            results[alg2] = []
    
    # Create temporary directory for results
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Common parameters
        env = Swimmer(horizon=horizon, gamma=1, render=False, clip=True)
        s_dim = env.state_dim
        a_dim = env.action_dim
        tot_params = s_dim * a_dim
        batch_size = 100
        lr = 1e-3
        
        for trial in range(start_trial, n_trials):
            print(f"Running trial {trial + 1}/{n_trials}")
            
            # Create trial directory
            trial_dir = os.path.join(temp_dir, f"trial_{trial}")
            os.makedirs(trial_dir, exist_ok=True)
            
            try:
                # Initialize policy
                pol = LinearGaussianPolicy(
                    parameters=np.zeros(tot_params),
                    dim_state=s_dim,
                    dim_action=a_dim,
                    std_dev=np.sqrt(1),
                    std_decay=0,
                    std_min=1e-5,
                    multi_linear=True,

                )

                # Run first algorithm
                if alg1 == 'pg':
                    alg = PolicyGradient(
                        lr=[lr],
                        lr_strategy="adam",
                        estimator_type="GPOMDP",
                        initial_theta=np.zeros(tot_params),
                        ite=ite,
                        batch_size=batch_size,
                        env=copy.deepcopy(env),
                        policy=copy.deepcopy(pol),
                        data_processor=IdentityDataProcessor(),
                        directory=os.path.join(trial_dir, alg1),
                        verbose=False,
                        natural=False,
                        checkpoint_freq=100,
                        n_jobs=n_workers,
                    )
                # Add other algorithm options here...
                
                alg.learn()
                results[alg1].append(alg.performance_idx)
                
                # Run second algorithm
                if alg2 == 'off_pg':
                    for window_length in window_lengths:
                        alg = OffPolicyGradient(
                            lr=[lr],
                            lr_strategy="adam",
                            initial_theta=np.zeros(tot_params),
                            ite=ite,
                            batch_size=batch_size,
                            env=copy.deepcopy(env),
                            policy=copy.deepcopy(pol),
                            data_processor=IdentityDataProcessor(),
                            directory=os.path.join(trial_dir, f"off_pg_{window_length}"),
                            verbose=False,
                            natural=False,
                            checkpoint_freq=100,
                            n_jobs=n_workers,
                            window_length=window_length,
                        )
                        alg.learn()
                        results[f'off_pg_{window_length}'].append(alg.performance_idx)
                else:
                    # Add handling for other algorithm types here
                    pass
                
                # Save checkpoint after each successful trial
                checkpoint_data = {
                    'results': results,
                    'last_trial': trial
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                    
            except Exception as e:
                print(f"Error in trial {trial}: {e}")
                # Save checkpoint before raising the exception
                checkpoint_data = {
                    'results': results,
                    'last_trial': trial - 1
                }
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                raise
            
    finally:
        shutil.rmtree(temp_dir)
        
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        
    return results

def plot_comparison(results, window=10, save_path='comparison_plot.png'):
    """
    Plot performance comparison with confidence intervals and save to PNG
    """
    plt.figure(figsize=(10, 6))
    
    # Get unique window lengths from results keys
    window_lengths = sorted([int(k.split('_')[-1]) for k in results.keys() if k != 'pg'])
    
    # Generate color map dynamically
    colors = {'pg': 'blue'}
    # Color map for off-policy variants
    color_map = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(window_lengths)))
    for i, w in enumerate(window_lengths):
        colors[f'off_pg_{w}'] = color_map[i]
    
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

def main():
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg1', type=str, default='pg', help='First algorithm to compare')
    parser.add_argument('--alg2', type=str, default='off_pg', help='Second algorithm to compare')
    parser.add_argument('--n_trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--ite', type=int, default=100, help='Number of iterations')
    parser.add_argument('--window_lengths', type=int, nargs='+', default=[3, 5, 10], 
                      help='Window lengths for off-policy (multiple values allowed)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                      help='Directory to store checkpoints')
    parser.add_argument("--horizon", help="The horizon amount.", type=int,default=100)
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()
    
    results = run_experiment(
        alg1=args.alg1,
        alg2=args.alg2,
        n_trials=args.n_trials,
        ite=args.ite,
        window_lengths=args.window_lengths,
        checkpoint_dir=args.checkpoint_dir,
        n_workers=args.n_workers,
        horizon=args.horizon
    )
    
    print("\nSaving comparison plot...")
    if args.alg2 == 'off_pg':
        plot_comparison(results, window=10, save_path=f'{args.alg1}_vs_off_pg_windows.png')
    else:
        plot_comparison(results, window=10, save_path=f'{args.alg1}_vs_{args.alg2}.png')

if __name__ == '__main__':
    main()
