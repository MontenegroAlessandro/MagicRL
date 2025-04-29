import json
import matplotlib.pyplot as plt
plt.style.use("ggplot")
import os
import numpy as np
import re

def plot_performance(subsample_rate=100):
    results_dir = "results/pendulum_MIS/sensitivity_trajectory/"
    # Create figure with higher DPI and larger size for better quality
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    # Set style for a cleaner look
    plt.style.use('bmh')
    
    # Define regex pattern
    pattern = r'(off_pg|pg)_(\d+)_([a-zA-Z0-9_]+)_(\d+)_adam_(\d+)_gaussian_batch_(\d+)_noclip(?:_window_(\d+)_(BH|MIS)_(\d+))?(?:_(\d+))?_var_(\d+)'
    
    
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
            algorithm = match.group(1)          # off_pg or pg
            iterations = match.group(2)         # iterations (5000)
            env_name = match.group(3)           # environment name (swimmer)
            env_param = match.group(4)          # number after env_name (200)
            adam_param = match.group(5)         # adam parameter (0001)
            batch_size = int(match.group(6))    # batch size (20)
            
            # Optional window parameters
            window_size = int(match.group(7)) if match.group(7) else 1
            sampling_type = match.group(8) if match.group(8) else None
            window_extra = match.group(9) if match.group(9) else None
            var_num = match.group(10)           # variant number (01)
            
            # Calculate total trajectories used per step: batch_size * window_size
            trajectories_per_step = batch_size * window_size
            
            # Create a clean label using the extracted information
            if algorithm != "N/A":
                clean_label = f"{algorithm} ("
                if batch_size != "N/A":
                    clean_label += f"Batch_size={batch_size}"
                if window_size > 1:
                    clean_label += f", Window={window_size}"
                if sampling_type != "N/A":
                    clean_label += f", {sampling_type}"
                clean_label += f", Total={trajectories_per_step})"
            
        else:
            # If regex doesn't match, use the original name
            clean_label = experiment_dir
            trajectories_per_step = 1  # Default if we can't extract info
            
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
                'std': std_perf,
                'trajectories_per_step': trajectories_per_step  # Store this for normalization
            }
    
    # Plot all data with consistent color scheme and sorted legend but with normalization by trajectories used
    for label, data in sorted(algorithm_data.items()):
        # Get the trajectories per step for normalization
        trajectories_per_step = data['trajectories_per_step']
        
        # Create normalized x-axis that accounts for number of trajectories used
        # This is the key change: we normalize by trajectories_per_step to compare fairly
        normalized_indices = np.arange(0, len(data['mean'])) * trajectories_per_step
        
        # Apply subsampling - take every subsample_rate points but maintain trajectory normalization
        subsample_mask = np.arange(0, len(normalized_indices), subsample_rate)
        subsampled_x = normalized_indices[subsample_mask]
        subsampled_mean = data['mean'][subsample_mask]
        subsampled_std = data['std'][subsample_mask]
        
        # Plot the subsampled data with normalized x-axis
        line = plt.plot(subsampled_x, subsampled_mean, label=label, linewidth=2)[0]
        
        # Add shaded region for variance using subsampled data
        plt.fill_between(
            subsampled_x,
            subsampled_mean - subsampled_std,
            subsampled_mean + subsampled_std,
            alpha=0.2,
            color=line.get_color()
        )
    
    plt.xlabel('Trajectories Used', fontsize=12)  # Changed from 'Episode' to 'Trajectories Used'
    plt.ylabel('Average Performance', fontsize=12)
    plt.title(f'Training Performance (Normalized by Batch Size × Window Size)', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Ensure the plot looks clean
    plt.tight_layout()
    
    # Save the plot as a PNG file
    save_path = os.path.join('visualizations', f'training_performance_normalized_by_trajectories.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Call with default subsampling rate of 50
    plot_performance(50)