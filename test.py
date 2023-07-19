from envs.gridworld_env import *
from envs.utils import *

horizon = 30
gamma = 1
grid_size = 20

"""
horizon: int = 0, 
gamma: float = 0, 
grid_size: int = 0,
reward_type: str = "linear", 
render: bool = False, 
dir: str = None, 
init_state: list = None,
obstacles: list = None
"""

square = Obstacle(
    type="square",
    features={"p1": Position(grid_size/2-1, grid_size/2+1),
              "p2": Position(grid_size/2+1, grid_size/2+1),
              "p3": Position(grid_size/2+1, grid_size),
              "p4": Position(grid_size/2-1, grid_size)}
)

env = GridWorldEnvCont(
    horizon=horizon, 
    gamma=gamma, 
    grid_size=grid_size, 
    reward_type="sparse", 
    render=True,
    dir="../../Desktop/cpgpe_exp/test",
    obstacles=[square],
    init_state=[1, 12]
)

if __name__ == "__main__":
    reward = 0
    obs = env.state.agent_pos
    for i in range(horizon):
        a = GWContMove(0.5, 30)
        pos, rew, abs = env.step(action=a)
        reward += rew
    env.reset()
    print(f"FINAL REWARD: {reward}")