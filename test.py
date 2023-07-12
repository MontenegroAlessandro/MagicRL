from envs import GridWorldEnv

env = GridWorldEnv(100, 0.99, 11, "sparse", "walls")

print(env.grid_map)
print(env.grid_view)