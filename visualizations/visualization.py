import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_performance():
    results_dir = "results"
    # Create figure with higher DPI and larger size for better quality
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Set style for a cleaner look
    plt.style.use('bmh')
    
    # Get all subdirectories in results
    for experiment_dir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, experiment_dir)
        if not os.path.isdir(experiment_path):
            continue
            
        # For each trial in the experiment
        performances = []
        for trial_dir in os.listdir(experiment_path):
            if not os.path.isdir(os.path.join(experiment_path, trial_dir)):
                continue
                
            # Read the results file
            results_file = os.path.join(experiment_path, trial_dir, "pg_results.json")
            if not os.path.exists(results_file):
                continue
                
            with open(results_file, 'r') as f:
                data = json.load(f)
                performances.append(data['performance_det'])
        
        if performances:
            performances = np.array(performances)  # Convert to numpy array for easier calculations
            # Calculate mean and std across trials
            mean_perf = np.mean(performances, axis=0)
            std_perf = np.std(performances, axis=0)
            
            # Plot mean line
            line = plt.plot(mean_perf, label=experiment_dir, linewidth=2)[0]
            # Add shaded region for variance (mean ± 1 standard deviation)
            plt.fill_between(
                range(len(mean_perf)),
                mean_perf - std_perf,
                mean_perf + std_perf,
                alpha=0.2,
                color=line.get_color()
            )
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Deterministic Performance', fontsize=12)
    plt.title('Training Performance Across Different Experiments', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Ensure the plot looks clean
    plt.tight_layout()
    
    # Save the plot as a PNG file
    save_path = os.path.join('visualizations', 'training_performance.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_performance()
