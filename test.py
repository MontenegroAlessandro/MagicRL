from envs.gridworld_env import *

horizon = 30
gamma = 1
grid_size = 20

env = GridWorldEnv(
    horizon=horizon, 
    gamma=gamma, 
    grid_size=grid_size, 
    reward_type="sparse", 
    env_type="walls",
    render=True,
    dir="test"
)

if __name__ == "__main__":
    reward = 0
    obs = env.state.agent_pos
    for i in range(horizon):
        if i % 2 == 0:
            if obs.x >= grid_size/2:
                a = "down"
            else:
                a = "up"
        else:
            if obs.y <= grid_size/2:
                a = "right"
            else:
                a = "left"
        pos, rew, abs = env.step(action=a)
        reward += rew
    env.reset()
    print(f"FINAL REWARD: {reward}")