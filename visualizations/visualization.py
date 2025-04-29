import json
import matplotlib.pyplot as plt
import os
import numpy as np
import re
import pandas as pd

def load_json_data(results_dir):
    """
    Load performance data from JSON files organized in the results directory.
    Returns a dictionary mapping experiment labels to performance statistics.
    """
    algorithm_data = {}
    
    # Define regex pattern for parsing directory names
    pattern = r'(off_pg|pg)_(\d+)_([a-zA-Z0-9_]+)_(\d+)_adam_(\d+)_gaussian_batch_(\d+)_noclip(?:_window_(\d+)_(BH|MIS)_(\d+))?(?:_(\d+))?_var_(\d+)'
    
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
            lr = match.group(5)                 # adam parameter (0001)
            batch_size = match.group(6)         # batch size (20)
            
            # Optional window parameters
            window_size = match.group(7) if match.group(7) else None
            sampling_type = match.group(8) if match.group(8) else None
            window_extra = match.group(9) if match.group(9) else None
            var_num = match.group(10)           # variant number (01)
            
            # Create a clean label using the extracted information
            clean_label = f"{algorithm} ("
            if batch_size:
                clean_label += f"Batch_size={batch_size}"
            if window_size:
                clean_label += f", Window={window_size}"
            if sampling_type:
                clean_label += f", {sampling_type}"
            if lr:
                clean_label += f", LR={lr}"
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
    
    return algorithm_data


def load_csv_data(base_dir, csv_folder):
    """
    Load performance data from all CSV files in the specified folder.
    Returns a dictionary mapping the folder name to performance statistics.
    """
    algorithm_data = {}
    
    csv_folder_path = os.path.join(base_dir, csv_folder)
    # Use the folder name as the label
    csv_label = csv_folder
    
    if not os.path.exists(csv_folder_path) or not os.path.isdir(csv_folder_path):
        print(f"Warning: CSV folder not found at {csv_folder_path}")
        return algorithm_data
    
    # Find all CSV files in the directory
    csv_files = [f for f in os.listdir(csv_folder_path) if f.lower().endswith('.csv')]
    
    if not csv_files:
        print(f"Warning: No CSV files found in {csv_folder_path}")
        return algorithm_data
    
    # List to store performance data from all CSV files
    all_csv_performances = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(csv_folder_path, csv_file)
        try:
            # Read the CSV file
            csv_data = pd.read_csv(csv_path)
            
            # Check if the 'Perf' column exists
            if 'Perf' in csv_data.columns:
                # Get the performance data
                csv_perf = csv_data['Perf'].values
                all_csv_performances.append(csv_perf)
            else:
                print(f"Warning: 'Perf' column not found in {csv_path}")
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
    
    if not all_csv_performances:
        print(f"Warning: No valid performance data found in CSV files in {csv_folder_path}")
        return algorithm_data
    
    # Find the minimum length to ensure all arrays have the same size
    min_length = min(len(perf) for perf in all_csv_performances)
    
    # Truncate all arrays to the minimum length
    all_csv_performances = [perf[:min_length] for perf in all_csv_performances]
    
    # Convert to numpy array for calculations
    all_csv_performances = np.array(all_csv_performances)
    
    # Calculate mean and std across all CSV files
    csv_mean = np.mean(all_csv_performances, axis=0)
    csv_std = np.std(all_csv_performances, axis=0)
    
    # Add to the algorithm_data dictionary
    algorithm_data[csv_label] = {
        'mean': csv_mean,
        'std': csv_std
    }
    
    return algorithm_data


def plot_performance(subsample_rate=1, specific_csv_folders=None):
    """
    Main function to plot performance data from both JSON and CSV sources.
    
    Parameters:
    - subsample_rate: Rate at which to sample data points for plotting
    - specific_csv_folders: List of specific folder names to include, or None to include all folders with CSVs
    """
    # Base directory containing all experiment data
    base_dir = "results/swimmer_MIS_nn/test/"
    
    # Load JSON data
    json_data = load_json_data(base_dir)
    
    # Initialize algorithm_data with JSON data
    algorithm_data = json_data
    
    # Find all folders that contain CSV files
    csv_folders = []
    
    if specific_csv_folders:
        # Use specific folders provided by the user
        csv_folders = specific_csv_folders
    else:
        # Find all folders in the base directory
        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            
            # Check if it's a directory
            if os.path.isdir(folder_path):
                # Look for CSV files in the folder
                csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
                
                # If CSV files exist, add this folder to the list
                if csv_files:
                    csv_folders.append(folder)
    
    # Load data from each CSV folder
    for csv_folder in csv_folders:
        csv_data = load_csv_data(base_dir, csv_folder)
        # Merge CSV data with existing data
        algorithm_data.update(csv_data)
    
    # Skip plotting if no data was found
    if not algorithm_data:
        print("No data found to plot.")
        return
    
    # Create figure with higher DPI and larger size for better quality
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    # Set style for a cleaner look
    plt.style.use('bmh')
    
    # Plot all data with consistent color scheme and sorted legend
    for label, data in sorted(algorithm_data.items()):
        # Apply subsampling - take every subsample_rate points
        x_indices = np.arange(0, len(data['mean']), subsample_rate)
        subsampled_mean = data['mean'][x_indices]
        subsampled_std = data['std'][x_indices]
        
        # Plot the subsampled data
        line = plt.plot(x_indices, subsampled_mean, label=label, linewidth=2)[0]
        
        # Add shaded region for variance using subsampled data
        plt.fill_between(
            x_indices,
            subsampled_mean - subsampled_std,
            subsampled_mean + subsampled_std,
            alpha=0.2,
            color=line.get_color()
        )
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Performance', fontsize=12)
    plt.title(f'Training Performance (Sampled every {subsample_rate} episodes)', fontsize=14, pad=20)

    plt.xlim(0, 300)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Ensure the plot looks clean
    plt.tight_layout()
    
    # Save the plot as a PNG file
    save_path = os.path.join('visualizations', f'training_performance_with_baseline.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.close()

if __name__ == "__main__":
    # Plot all folders containing CSV files
    plot_performance(subsample_rate=1)
    
    # Alternatively, you can specify which folders to include:
    # plot_performance(specific_csv_folders=["baseline_storm_pg", "another_baseline_folder"])