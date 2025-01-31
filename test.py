import numpy as np
import timeit

def setup():
    # Assuming typical dimensions from your code:
    # num_updates = window_length (e.g., 5)
    # num_trajectories = window_size (e.g., window_length * batch_size = 5 * 10 = 50)
    global products
    products = np.random.random((5, 50))

def method1_numpy():
    # Standard NumPy operation
    products[:-1, :-10] = products[1:, 10:]

def method2_data():
    # Direct data manipulation (unsafe)
    products.data = products[1:, 10:].data
    products.resize((5, 50))

# Time each method
time1 = timeit.timeit(method1_numpy, setup=setup, number=10000)
time2 = timeit.timeit(method2_data, setup=setup, number=10000)

print(f"NumPy operation time: {time1:.6f} seconds")
print(f"Data manipulation time: {time2:.6f} seconds")