# Environments :earth_africa:
In this folder you have a lot of environments:
- Ant by `MuJoCo`
- Cartpole by `MagicRL`
- Grid World by `MagicRL`
- Half Cheetah by `MuJoCo`
- Hopper by `MuJoCo`
- Humanoid by `MuJoCo`
- LQR by `MagicRL`
- Reacher by `MuJoCo`
- Swimmer by `MuJoCo`

The interface is the classic one, so each step of an environment returns:
- next **state**
- instantaneous **reward**
- flag specifying whether the new state is **absorbing** 
- **info** dictionary

---

# Environments with Costs :money_mouth:
Each environment class has a flag called `with_costs(=False)` and has a cost counter field called
`how_many_costs(=0)`.
We have some Environments with the costs:
- LQR by `MagicRL`

When you are using an environment with its cost function, you can find the cost values in 
`info["costs"]` in which is stored a list of costs.