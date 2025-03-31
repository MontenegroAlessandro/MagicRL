import json
import matplotlib.pyplot as plt
import os
import numpy as np
import re

def extract_clean_label(folder_name):
    pattern = r'(off_pg|pg)_\d+_pendulum_\d+_adam_\d+_gaussian_batch_(\d+)_noclip(?:_window_(\d+)_?(BH|MIS))?_(\d+)_var_\d+'
    match = re.search(pattern, folder_name)
    
    if match:
        algorithm = match.group(1)
        batch_size = int(match.group(2))
        window = match.group(3) if match.group(3) else "N/A"
        bh_or_mis = match.group(4) if match.group(4) else "N/A"
        
        # Create a clean label
        if algorithm != "N/A":
            clean_label = f"{algorithm} ("
            if batch_size != "N/A":
                clean_label += f"Batch_size={batch_size}"
            if window != "N/A":
                clean_label += f", Window={window}"
            if bh_or_mis != "N/A":
                clean_label += f", {bh_or_mis}"
            clean_label += ")"
            
        # Only return the clean label
        return clean_label, batch_size
    return None

def plot_performance_by_trajectory():
    results_dir = "results/pendulum_MIS"
    plt.figure(figsize=(12, 8), dpi=300)
    plt.style.use('bmh')
    
    experiment_data = {}
    
    for experiment_dir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, experiment_dir)
        if not os.path.isdir(experiment_path):
            continue
        
        # Extract clean label using the regex pattern
        clean_label, batch_size = extract_clean_label(experiment_dir)
        if clean_label is None:
            continue
        
        # Extract batch size directly from the folder name for grouping
        
        if batch_size not in experiment_data:
            experiment_data[batch_size] = []
        
        performances = []
        
        for trial_dir in os.listdir(experiment_path):
            trial_path = os.path.join(experiment_path, trial_dir)
            if not os.path.isdir(trial_path):
                continue
                
            results_file = os.path.join(trial_path, "pg_results.json")
            if not os.path.exists(results_file):
                continue
                
            with open(results_file, 'r') as f:
                data = json.load(f)
                performances.append(data['performance'])
                
        if performances:
            performances = np.array(performances)
            mean_perf = np.mean(performances, axis=0)
            std_perf = np.std(performances, axis=0)
            experiment_data[batch_size].append((clean_label, mean_perf, std_perf))
    
    # Sort within each batch size group to ensure consistent color ordering
    for batch_size in experiment_data:
        experiment_data[batch_size].sort(key=lambda x: x[0])
    
    # Plot the data
    for batch_size, experiments in sorted(experiment_data.items()):
        for clean_label, mean_perf, std_perf in experiments:
            effective_x = np.arange(0, len(mean_perf) * batch_size, batch_size)
            
            # Use the clean_label directly for the plot legend
            line = plt.plot(effective_x, mean_perf, label=clean_label, linewidth=2)[0]
            plt.fill_between(
                effective_x,
                mean_perf - std_perf,
                mean_perf + std_perf,
                alpha=0.2,
                color=line.get_color()
            )
    
    plt.xlabel('Collected Trajectories', fontsize=12)
    plt.ylabel('Average Performance', fontsize=12)
    plt.title('Trajectory Equalized Performance: PG vs Off-PG in Inverted Pendulum', fontsize=14, pad=20)
    
    # Improve legend with sorting and formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_path = os.path.join('visualizations', 'performance_by_trajectory.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_performance_by_trajectory()