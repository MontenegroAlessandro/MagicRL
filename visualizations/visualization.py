import json
import matplotlib.pyplot as plt
import os
import numpy as np
import re

def plot_performance():
    results_dir = "results/pendulum_comparison_5batch/"
    # Create figure with higher DPI and larger size for better quality
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Set style for a cleaner look
    plt.style.use('bmh')
    
    # Define regex pattern
    pattern = r'(off_pg|pg)_\d+_pendulum_\d+_adam_\d+_gaussian_batch_(\d+)_noclip(?:_window_(\d+)_?(BH|MIS))?_(\d+)_var_\d+'
    
    # Dictionary to store grouped data for legend organization
    algorithm_data = {}
    
    # Get all subdirectories in results
    for experiment_dir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, experiment_dir)

        if not os.path.isdir(experiment_path):
            continue
        
        # Extract info using regex
        match = re.search(pattern, experiment_dir)
        if match:
            algorithm = match.group(1)
            batch_size = match.group(2)
            window = match.group(3) if match.group(3) else "N/A"
            bh_or_mis = match.group(4) if match.group(4) else "N/A"
            
            # Create a clean label using the extracted information
            if algorithm != "N/A":
                clean_label = f"{algorithm} ("
                if batch_size != "N/A":
                    clean_label += f"Batch_size={batch_size}"
                if window != "N/A":
                    clean_label += f", Window={window}"
                if bh_or_mis != "N/A":
                    clean_label += f", {bh_or_mis}"
                clean_label += ")"
            
        else:
            # If regex doesn't match, use the original name
            clean_label = experiment_dir
            
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
                performances.append(data['performance'])
        
        if performances:
            performances = np.array(performances)  # Convert to numpy array for easier calculations
            # Calculate mean and std across trials
            mean_perf = np.mean(performances, axis=0)
            std_perf = np.std(performances, axis=0)
            
            # Store the processed data with the clean label
            algorithm_data[clean_label] = {
                'mean': mean_perf,
                'std': std_perf
            }
    
    # Plot all data with consistent color scheme and sorted legend
    for label, data in sorted(algorithm_data.items()):
        line = plt.plot(data['mean'], label=label, linewidth=2)[0]
        # Add shaded region for variance
        plt.fill_between(
            range(len(data['mean'])),
            data['mean'] - data['std'],
            data['mean'] + data['std'],
            alpha=0.2,
            color=line.get_color()
        )
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Performance', fontsize=12)
    plt.title('Training Performance Across Different Algorithms', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Ensure the plot looks clean
    plt.tight_layout()
    
    # Save the plot as a PNG file
    save_path = os.path.join('visualizations', 'training_performance.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_performance()