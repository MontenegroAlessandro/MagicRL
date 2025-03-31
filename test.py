import re

pattern = r'(off_pg|pg)_\d+_pendulum_\d+_adam_\d+_gaussian_batch_(\d+)_noclip(?:_window_(\d+)_?(BH|MIS))?_(\d+)_var_\d+'

# Example usage
filenames = [
    "off_pg_2000_pendulum_200_adam_0001_gaussian_batch_5_noclip_window_16_BH_4_var_01",
    "off_pg_2000_pendulum_200_adam_0001_gaussian_batch_5_noclip_window_16_MIS_4_var_01",
    "pg_2000_pendulum_200_adam_0001_gaussian_batch_5_noclip_4_var_01"
]

for filename in filenames:
    match = re.search(pattern, filename)
    if match:
        algorithm = match.group(1)
        batch_size = match.group(2)
        window = match.group(3) if match.group(3) else "N/A"
        bh_or_mis = match.group(4) if match.group(4) else "N/A"
        
        print(f"Filename: {filename}")
        print(f"  Algorithm: {algorithm}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Window Size: {window}")
        print(f"  BH or MIS: {bh_or_mis}")
        print()