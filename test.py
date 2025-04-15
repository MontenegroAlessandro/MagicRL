import numpy as np
import hashlib
import timeit

# Define the two operations as functions

def hash_method(parameters: np.ndarray) -> str:
    """Convert parameters to bytes and return their MD5 hash (as hex string)."""
    param_bytes = parameters.tobytes()
    return hashlib.md5(param_bytes).hexdigest()

def tuple_conversion(state: np.ndarray) -> tuple:
    """Convert a NumPy array (or a list of parameters) to a tuple."""
    return tuple(state)

# Set up input arrays (simulate a long list of parameters and state values)
dim = 10000  # You can change this value to simulate longer or shorter vectors.
parameters = np.random.randn(dim)
state = np.random.randn(dim)

# Set the number of iterations for the timeit measurement
iterations = 1000

# Measure the execution time for the hash method
hash_time = timeit.timeit(lambda: hash_method(parameters), number=iterations)

# Measure the execution time for converting a long list to a tuple
tuple_time = timeit.timeit(lambda: tuple_conversion(state), number=iterations)

print(f"Over {iterations} iterations:")
print(f"  Hash method (MD5 after bytes conversion): {hash_time:.6f} seconds total")
print(f"  Tuple conversion: {tuple_time:.6f} seconds total")
