import numpy as np
import torch
import time
import platform

def test_speed_comparison():
    """Compare speed of NumPy vs PyTorch implementations of the given function."""
    
    # Print system information
    print("System Information:")
    print(f"Python: {platform.python_version()}")
    print(f"NumPy: {np.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("\n")
    
    # Test configurations - different sizes to test scaling
    configs = [
        {"name": "Small", "num_trajectories": 100, "timesteps": 20, "action_dim": 2, "batch_size": 5, "runs": 100},
        {"name": "Medium", "num_trajectories": 500, "timesteps": 50, "action_dim": 4, "batch_size": 10, "runs": 50},
        {"name": "Large", "num_trajectories": 1000, "timesteps": 100, "action_dim": 8, "batch_size": 20, "runs": 10}
    ]
    
    for config in configs:
        # Unpack config
        name = config["name"]
        num_trajectories = config["num_trajectories"]
        timesteps = config["timesteps"]
        action_dim = config["action_dim"]
        batch_size = config["batch_size"]
        runs = config["runs"]
        alpha = 0.5
        
        print(f"\nRunning {name} test: Shape ({num_trajectories}, {timesteps}, {action_dim})")
        
        # Generate test data
        np_current_means = np.random.randn(num_trajectories, timesteps, action_dim)
        np_past_means = np.random.randn(num_trajectories, timesteps, action_dim)
        np_var = 0.1
        
        # Create PyTorch tensors from the same data
        torch_current_means = torch.tensor(np_current_means, dtype=torch.float32)
        torch_past_means = torch.tensor(np_past_means, dtype=torch.float32)
        torch_var = torch.tensor(np_var, dtype=torch.float32)
        
        # Test NumPy implementation
        np_start = time.time()
        for _ in range(runs):
            # NumPy implementation
            mean_diff = np_current_means - np_past_means
            timestep_divergence = np.exp(- alpha * (1 - alpha) * np.sum(mean_diff * mean_diff, axis=2) / 2 * np_var)
            trajectories_divergence = np.prod(timestep_divergence, axis=1).reshape(-1, batch_size)
            np_result = np.mean(trajectories_divergence, axis=1)
        np_time = (time.time() - np_start) / runs
        
        # Test PyTorch implementation (CPU)
        torch_cpu_start = time.time()
        for _ in range(runs):
            # PyTorch implementation
            mean_diff = torch_current_means - torch_past_means
            timestep_divergence = torch.exp(- alpha * (1 - alpha) * torch.sum(mean_diff * mean_diff, dim=2) / 2 * torch_var)
            trajectories_divergence = torch.prod(timestep_divergence, dim=1).reshape(-1, batch_size)
            torch_cpu_result = torch.mean(trajectories_divergence, dim=1)
        torch_cpu_time = (time.time() - torch_cpu_start) / runs
        
        # Print CPU results
        np_time_ms = np_time * 1000
        torch_cpu_time_ms = torch_cpu_time * 1000
        cpu_speedup = np_time / torch_cpu_time
        
        print(f"NumPy time: {np_time_ms:.2f} ms")
        print(f"PyTorch CPU time: {torch_cpu_time_ms:.2f} ms")
        print(f"CPU speedup: {cpu_speedup:.2f}x {'(PyTorch faster)' if cpu_speedup > 1 else '(NumPy faster)'}")
        
        # Check output differences
        torch_cpu_result_np = torch_cpu_result.numpy()
        output_diff = np.abs(np_result - torch_cpu_result_np).mean()
        print(f"Output difference: {output_diff:.9f}")
        
        # Test PyTorch implementation (GPU) if available
        if torch.cuda.is_available():
            # Move data to GPU
            torch_gpu_current_means = torch_current_means.cuda()
            torch_gpu_past_means = torch_past_means.cuda()
            torch_gpu_var = torch_var.cuda()
            
            # Warm-up run
            mean_diff = torch_gpu_current_means - torch_gpu_past_means
            timestep_divergence = torch.exp(- alpha * (1 - alpha) * torch.sum(mean_diff * mean_diff, dim=2) / 2 * torch_gpu_var)
            trajectories_divergence = torch.prod(timestep_divergence, dim=1).reshape(-1, batch_size)
            _ = torch.mean(trajectories_divergence, dim=1)
            
            # Timing run
            torch.cuda.synchronize()  # Ensure previous operations are completed
            torch_gpu_start = time.time()
            for _ in range(runs):
                mean_diff = torch_gpu_current_means - torch_gpu_past_means
                timestep_divergence = torch.exp(- alpha * (1 - alpha) * torch.sum(mean_diff * mean_diff, dim=2) / 2 * torch_gpu_var)
                trajectories_divergence = torch.prod(timestep_divergence, dim=1).reshape(-1, batch_size)
                torch_gpu_result = torch.mean(trajectories_divergence, dim=1)
                torch.cuda.synchronize()  # Ensure each iteration is completed
            torch_gpu_time = (time.time() - torch_gpu_start) / runs
            
            # Print GPU results
            torch_gpu_time_ms = torch_gpu_time * 1000
            gpu_speedup = np_time / torch_gpu_time
            
            print(f"PyTorch GPU time: {torch_gpu_time_ms:.2f} ms")
            print(f"GPU speedup: {gpu_speedup:.2f}x")
            
            # Check output differences
            torch_gpu_result_np = torch_gpu_result.cpu().numpy()
            gpu_output_diff = np.abs(np_result - torch_gpu_result_np).mean()
            print(f"GPU output difference: {gpu_output_diff:.9f}")
    
    print("\nSummary: When to use PyTorch vs NumPy")
    print("1. Use PyTorch when:")
    print("   - You need automatic differentiation (gradients)")
    print("   - You have access to GPU acceleration")
    print("   - Your models will be part of a larger deep learning framework")
    print("   - You're working with batched operations on large datasets")
    print("2. Use NumPy when:")
    print("   - You only need numerical computations without gradients")
    print("   - Your code is simpler and doesn't need GPU acceleration")
    print("   - You're working with smaller datasets or non-ML applications")
    print("   - You need specialized scientific functions not available in PyTorch")

if __name__ == "__main__":
    test_speed_comparison()