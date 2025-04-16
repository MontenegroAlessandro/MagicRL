import numpy as np
import torch
import time

def test_conversion_speed(sizes=[1000, 10000, 100000, 1000000], iterations=10):
    """
    Test the speed difference between:
    1. torch.from_numpy(x).float().to(device)
    2. torch.from_numpy(x).float() (without device specification)
    3. torch.tensor(x)
    
    Only uses CPU as requested.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device("cpu")
    
    # Warm-up
    x_warmup = np.random.rand(1000).astype(np.float32)
    for _ in range(5):
        _ = torch.from_numpy(x_warmup).float().to(device)
        _ = torch.from_numpy(x_warmup).float()
        _ = torch.tensor(x_warmup)
    
    print("Comparing NumPy to PyTorch conversion speeds:")
    print(f"{'Array Size':<12} {'from_numpy+device':<20} {'from_numpy only':<20} {'torch.tensor':<20}")
    print("-" * 70)
    
    for size in sizes:
        # Create a NumPy array
        x = np.random.rand(size).astype(np.float32)
        
        # Time torch.from_numpy with device
        start = time.perf_counter()
        for _ in range(iterations):
            t1 = torch.from_numpy(x).float().to(device)
            _ = t1.sum().item()  # Ensure computation completes
        from_numpy_device_time = (time.perf_counter() - start) / iterations * 1000  # ms
        
        # Time torch.from_numpy without device
        start = time.perf_counter()
        for _ in range(iterations):
            t2 = torch.from_numpy(x).float()
            _ = t2.sum().item()  # Ensure computation completes
        from_numpy_time = (time.perf_counter() - start) / iterations * 1000  # ms
        
        # Time torch.tensor
        start = time.perf_counter()
        for _ in range(iterations):
            t3 = torch.tensor(x)
            _ = t3.sum().item()  # Ensure computation completes
        tensor_time = (time.perf_counter() - start) / iterations * 1000  # ms
        
        print(f"{size:<12} {from_numpy_device_time:.4f} ms {'':<5} {from_numpy_time:.4f} ms {'':<5} {tensor_time:.4f} ms")
    
    print("\nConclusions:")
    print("1. torch.from_numpy() without device specification is typically fastest for CPU operations")
    print("2. Adding .to(device) when device=cpu adds a small overhead")
    print("3. torch.tensor() is usually slower as it always creates a copy of the data")

if __name__ == "__main__":
    test_conversion_speed()