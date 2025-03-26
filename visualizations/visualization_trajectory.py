import json
import matplotlib.pyplot as plt
import os
import numpy as np
import re

def extract_batch_size(folder_name):
    match = re.search(r'batch_(\d+)', folder_name)
    return int(match.group(1)) if match else None

def plot_performance_by_trajectory():
    results_dir = "results/cheetah"
    plt.figure(figsize=(12, 8), dpi=300)
    plt.style.use('bmh')
    
    experiment_data = {}
    
    for experiment_dir in os.listdir(results_dir):
        experiment_path = os.path.join(results_dir, experiment_dir)
        if not os.path.isdir(experiment_path):
            continue
        
        batch_size = extract_batch_size(experiment_dir)
        if batch_size is None:
            continue
        
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
                performances.append(data['performance_det'])
                
        if performances:
            performances = np.array(performances)
            mean_perf = np.mean(performances, axis=0)
            std_perf = np.std(performances, axis=0)
            experiment_data[batch_size].append((experiment_dir, mean_perf, std_perf))
    
    for batch_size, experiments in experiment_data.items():
        for experiment_name, mean_perf, std_perf in experiments:
            effective_x = np.arange(0, len(mean_perf) * batch_size, batch_size)
            
            line = plt.plot(effective_x, mean_perf, label=experiment_name, linewidth=2)[0]
            plt.fill_between(
                effective_x,
                mean_perf - std_perf,
                mean_perf + std_perf,
                alpha=0.2,
                color=line.get_color()
            )
    
    plt.xlabel('Collected Trajectories', fontsize=12)
    plt.ylabel('Average Deterministic Performance', fontsize=12)
    plt.title('Trajectory equalized performance: pg vs off_pg in inverted_pendulum', fontsize=14, pad=20)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join('visualizations', 'performance_by_trajectory.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_performance_by_trajectory()
