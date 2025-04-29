import json
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import pandas as pd

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
            
        # Return both the clean label and batch size
        return clean_label, batch_size
    return None, None

def calculate_trajectories(algorithm_type, num_episodes, params):
    """
    Calculate cumulative trajectory counts for different algorithm types
    
    Parameters:
    - algorithm_type: 'storm-pg', 'svrpg', or 'srvrpg'
    - num_episodes: Number of episodes/data points
    - params: Dictionary with algorithm-specific parameters:
        - For storm-pg: {'N': initial_batch, 'B': regular_batch}
        - For svrpg/srvrpg: {'N': outer_batch, 'B': inner_batch, 'X': inner_loop_count}
    
    Returns:
    - cumulative_trajectories: List of cumulative trajectory counts
    
    Raises:
    - KeyError: If a required parameter is missing
    """
    if algorithm_type.lower() == 'storm-pg':
        # N trajectories first iteration, B every time after
        # Require explicit parameters - no defaults
        if 'N' not in params or 'B' not in params:
            raise KeyError(f"Missing required parameters for {algorithm_type}. Need 'N' and 'B'.")
            
        N = params['N']
        B = params['B']
        
        trajectories = [N if i == 0 else B for i in range(num_episodes)]
        
    elif algorithm_type.lower() in ['svrpg', 'srvrpg']:
        # N trajectories, then B trajectories X times, repeat
        # Require explicit parameters - no defaults
        if 'N' not in params or 'B' not in params or 'X' not in params:
            raise KeyError(f"Missing required parameters for {algorithm_type}. Need 'N', 'B', and 'X'.")
            
        N = params['N']
        B = params['B']
        X = params['X']
        
        trajectories = []
        while len(trajectories) < num_episodes:
            trajectories.append(N)
            for _ in range(X):
                if len(trajectories) < num_episodes:
                    trajectories.append(B)
    else:
        # For unknown algorithm types, require at least B
        if 'B' not in params:
            raise KeyError(f"Missing required parameter 'B' for {algorithm_type}.")
            
        B = params['B']
        trajectories = [B] * num_episodes
    
    # Calculate cumulative sum
    cumulative = np.cumsum(trajectories).tolist()
    
    # Truncate to requested number of episodes
    return cumulative[:num_episodes]

def load_json_data(results_dir):
    """
    Load performance data from JSON files in the results directory.
    Returns a dictionary mapping experiment labels to performance statistics and batch sizes.
    """
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
            experiment_data[batch_size].append((clean_label, mean_perf, std_perf, batch_size))
    
    return experiment_data

def load_csv_data(base_dir, folder, algorithm_type, trajectory_params):
    """
    Load performance data from all CSV files in the specified folder.
    Apply trajectory-based x-axis calculation based on algorithm type.
    
    Parameters:
    - base_dir: Base directory containing the CSV folders
    - folder: Name of folder containing CSV files
    - algorithm_type: Algorithm type for trajectory calculation ('storm-pg', 'svrpg', 'srvrpg')
    - trajectory_params: Dictionary with parameters for trajectory calculation
    
    Returns:
    - Dictionary with batch_size as key and list of (label, mean, std, batch_size) tuples as value
    """
    csv_data = {}
    csv_folder_path = os.path.join(base_dir, folder)
    
    # Use the folder name as the label
    label = folder
    
    if not os.path.exists(csv_folder_path) or not os.path.isdir(csv_folder_path):
        print(f"Warning: CSV folder not found at {csv_folder_path}")
        return csv_data
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_folder_path) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"Warning: No CSV files found in {csv_folder_path}")
        return csv_data
    
    # List to store performance data from all CSV files
    all_csv_performances = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder_path, csv_file)
        try:
            # Read the CSV file
            csv_data_df = pd.read_csv(csv_path)
            
            # Check if the 'Perf' column exists
            if 'Perf' in csv_data_df.columns:
                # Get the performance data
                csv_perf = csv_data_df['Perf'].values
                all_csv_performances.append(csv_perf)
            else:
                print(f"Warning: 'Perf' column not found in {csv_path}")
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
    
    if not all_csv_performances:
        print(f"Warning: No valid performance data found in CSV files in {csv_folder_path}")
        return csv_data
    
    # Find the minimum length to ensure all arrays have the same size
    min_length = min(len(perf) for perf in all_csv_performances)
    
    # Truncate all arrays to the minimum length
    all_csv_performances = [perf[:min_length] for perf in all_csv_performances]
    
    # Convert to numpy array for calculations
    all_csv_performances = np.array(all_csv_performances)
    
    # Calculate mean and std across all CSV files
    csv_mean = np.mean(all_csv_performances, axis=0)
    csv_std = np.std(all_csv_performances, axis=0)
    
    # Use provided batch size from trajectory params
    batch_size = trajectory_params['B']  # Will raise KeyError if missing
    
    # Add to the csv_data dictionary
    if batch_size not in csv_data:
        csv_data[batch_size] = []
    
    csv_data[batch_size].append((label, csv_mean, csv_std, batch_size))
    
    return csv_data

def plot_performance_by_trajectory(csv_folders=None):
    """
    Plot performance data by trajectory, combining JSON and CSV data sources.
    
    Parameters:
    - csv_folders: Dictionary mapping folder names to (algorithm_type, params) tuples
                  Example: {'storm_baseline': ('storm-pg', {'N': 100, 'B': 20})}
    """
    results_dir = "results/pendulum_MIS/sensitivity_trajectory2"
    plt.figure(figsize=(12, 8), dpi=300)
    plt.style.use('bmh')
    
    # Load JSON data
    experiment_data = load_json_data(results_dir)
    
    # Load CSV data if specified
    if csv_folders:
        for folder, (algorithm_type, params) in csv_folders.items():
            csv_data = load_csv_data(results_dir, folder, algorithm_type, params)
            
            # Merge CSV data with JSON data
            for batch_size, entries in csv_data.items():
                if batch_size not in experiment_data:
                    experiment_data[batch_size] = []
                experiment_data[batch_size].extend(entries)
    
    # Sort within each batch size group to ensure consistent color ordering
    for batch_size in experiment_data:
        experiment_data[batch_size].sort(key=lambda x: x[0])
    
    # Plot the data
    for batch_size, experiments in sorted(experiment_data.items()):
        for clean_label, mean_perf, std_perf, actual_batch in experiments:
            # Check if this is CSV data (based on the label matching a folder name)
            is_csv_data = csv_folders and clean_label in csv_folders
            
            if is_csv_data:
                # Use the exact trajectory parameters provided for CSV data
                algorithm_type, params = csv_folders[clean_label]
                # Calculate cumulative trajectories based on exact algorithm parameters
                x_values = calculate_trajectories(
                    algorithm_type, 
                    len(mean_perf), 
                    params
                )
            else:
                # For JSON data, use the standard approach from the original code
                effective_x = np.arange(0, len(mean_perf) * batch_size, batch_size)
                x_values = effective_x
            
            # Ensure x_values and mean_perf have the same length
            x_values = x_values[:len(mean_perf)]
            
            # Use the clean_label directly for the plot legend
            line = plt.plot(x_values, mean_perf, label=clean_label, linewidth=2)[0]
            plt.fill_between(
                x_values,
                mean_perf - std_perf,
                mean_perf + std_perf,
                alpha=0.2,
                color=line.get_color()
            )
    
    plt.xlabel('Collected Trajectories', fontsize=12)
    plt.ylabel('Average Performance', fontsize=12)
    plt.title('Trajectory Equalized Performance: PG vs Off-PG in Inverted Pendulum', fontsize=14, pad=20)

    plt.xlim(0, 5000)  # Adjusted x-axis limit for better visibility
    
    # Improve legend with sorting and formatting
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles)))
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    save_path = os.path.join('visualizations', 'performance_by_trajectory_with_baselines.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Define CSV folders with their algorithm types and parameters
    csv_folders = {
        'storm-pg_25N_5B': ('storm-pg', {'N': 25, 'B': 5}),
        'svrpg_100N_10B': ('svrpg', {'N': 100, 'B': 10, 'X': 10}),
        'svrpg_50N_5B': ('svrpg', {'N': 50, 'B': 5, 'X': 10}),
        'srvrpg_50N_5B': ('svrpg', {'N': 50, 'B': 5, 'X': 10}),
        'srvrpg_100N_10B': ('svrpg', {'N': 100, 'B': 10, 'X': 10})
        # Add more folders as needed
    }
    
    plot_performance_by_trajectory(csv_folders)