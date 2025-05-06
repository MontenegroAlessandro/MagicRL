import json
import matplotlib.pyplot as plt
import os
import numpy as np
import re

def plot_combined_performance(subsample_rate=50):
    results_dir = "results/cartpole/sensitivity_trajectory2/"
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    # Set style for a cleaner look
    plt.style.use('bmh')
    
    # Define a single regex pattern that works for both plots
    pattern = r'(off_pg|pg)_(\d+)_([a-zA-Z0-9_]+)_(\d+)_adam_(\d+)_gaussian_batch_(\d+)_noclip(?:_window_(\d+)_(BH|MIS)(?:_(\d+))?)?(?:_(\d+))?_var_(\d+)'
    
    # Create a master dictionary to store ALL experiment data for both plots
    all_experiments = {}
    
    # Process each experiment directory only ONCE
    for experiment_dir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, experiment_dir)
        if not os.path.isdir(experiment_path):
            continue
        
        # Extract info using regex
        match = re.search(pattern, experiment_dir)
        if not match:
            continue
            
        algorithm = match.group(1)          # off_pg or pg
        batch_size = int(match.group(6))    # batch size (20)
        window_size = int(match.group(7)) if match.group(7) else 1
        sampling_type = match.group(8) if match.group(8) else None
        
        # Calculate total trajectories used per step: batch_size * window_size
        trajectories_per_step = batch_size * window_size
        
        # Create a STANDARDIZED label that will be used in both plots
        clean_label = f"{algorithm} ("
        clean_label += f"Batch={batch_size}"
        if window_size > 1:
            clean_label += f", Window={window_size}"
        if sampling_type:
            clean_label += f", {sampling_type}"
        clean_label += f")"
        
        # Collect all trial performances
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
            performances = np.array(performances)
            mean_perf = np.mean(performances, axis=0)
            std_perf = np.std(performances, axis=0)
            
            # Store ALL the data needed for both plots
            all_experiments[clean_label] = {
                'mean': mean_perf,
                'std': std_perf,
                'batch_size': batch_size,
                'window_size': window_size,
                'trajectories_per_step': trajectories_per_step,
                'algorithm': algorithm,
                'sampling_type': sampling_type
            }
    
    # Create a color map based on unique labels
    import matplotlib.cm as cm
    # Sort labels to ensure consistent color assignment
    sorted_labels = sorted(all_experiments.keys())
    colors = cm.tab10(np.linspace(0, 1, len(sorted_labels)))
    color_map = {label: colors[i] for i, label in enumerate(sorted_labels)}
    
    # ======== SUBPLOT 1: First plot (normalized trajectories) ========
    for label, data in sorted(all_experiments.items()):
        # Get the trajectories per step for normalization
        trajectories_per_step = data['trajectories_per_step']
        
        # Create normalized x-axis that accounts for number of trajectories used
        normalized_indices = np.arange(0, len(data['mean'])) * trajectories_per_step
        
        # Apply subsampling
        subsample_mask = np.arange(0, len(normalized_indices), subsample_rate)
        subsampled_x = normalized_indices[subsample_mask]
        subsampled_mean = data['mean'][subsample_mask]
        subsampled_std = data['std'][subsample_mask]
        
        # Plot with the assigned color from color_map
        color = color_map[label]
        line = ax1.plot(subsampled_x, subsampled_mean, linewidth=2, color=color)[0]
        
        # Add shaded region for variance
        ax1.fill_between(
            subsampled_x,
            subsampled_mean - subsampled_std,
            subsampled_mean + subsampled_std,
            alpha=0.2,
            color=color
        )
    
    ax1.set_xlabel('Trajectories Used', fontsize=12)
    ax1.set_ylabel('Average Performance', fontsize=12)
    ax1.set_title('Training Performance (Normalized by Batch Size × Window Size)', fontsize=14, pad=20)
    ax1.grid(True, alpha=0.3)
    
    # ======== SUBPLOT 2: Second plot (trajectory equalized) ========
    # Group experiments by batch size for the second plot
    batch_grouped_data = {}
    for label, data in all_experiments.items():
        batch_size = data['batch_size']
        if batch_size not in batch_grouped_data:
            batch_grouped_data[batch_size] = []
        batch_grouped_data[batch_size].append((label, data))
    
    # Plot the data for second subplot
    for batch_size, experiment_group in sorted(batch_grouped_data.items()):
        for label, data in sorted(experiment_group):
            # Calculate x-axis values using batch_size
            effective_x = np.arange(0, len(data['mean']) * batch_size, batch_size)
            
            # Use exactly the same color from the color_map
            color = color_map[label]
            line = ax2.plot(effective_x, data['mean'], linewidth=2, color=color)[0]
            
            ax2.fill_between(
                effective_x,
                data['mean'] - data['std'],
                data['mean'] + data['std'],
                alpha=0.2,
                color=color
            )
    
    ax2.set_xlabel('Collected Trajectories', fontsize=12)
    ax2.set_ylabel('Average Performance', fontsize=12)
    ax2.set_title('Trajectory Equalized Performance: PG vs Off-PG in Cartpole', fontsize=14, pad=20)
    ax2.set_xlim(0, 20000)
    ax2.grid(True, alpha=0.3)
    
    # Create a single legend for the entire figure
    handles = []
    labels = []
    for label in sorted_labels:
        color = color_map[label]
        line = plt.Line2D([0], [0], color=color, linewidth=2)
        handles.append(line)
        labels.append(label)
    
    # Position the legend below both subplots
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.00),
              ncol=3, fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    # Adjust figure layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make space for the legend
    
    # Save the plot
    save_path = os.path.join('visualizations', 'combined_performance_plots.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    
    plt.close()

if __name__ == "__main__":
    # Call with default subsampling rate of 50
    plot_combined_performance(50)