# MagicRL :crystal_ball: :rocket:
Magic as the AS Roma! :wolf:

## Set up the environment.
(Attention is all) You need ( :joy: ) an `anaconda3` environment with python 3.11.5.
```bash
conda create --name name python=3.11.5
conda activate name
```

Install the packages.
```bash
pip3 install -r requirements.txt
```

## Run experiments.
All you need is in `run.py`, which requires several parameters:
- "--dir": specifies the directory in which will be saved the results;
- "--ite": how many iterations the algorithm must do;
- "--alg": the algorithm to run, you can select "pg" or "pgpe";
- "--var": the exploration amount, it is $\sigma^2$;
- "--pol": the policy to use, you can select "linear" or "nn";
- "--env": the environment on which the learning has to be done, you can select "swimmer", "half_cheetah", "hopper";
- "--horizon": set the horizon of the problem;
- "--gamma": set the discount factor of the problem;
- "--lr": set the step size;
- "--lr_strategy": set the learning rate schedule, you can select "constant" or "adam";
- "--n_workers": specifies how many trajectories are evaluated in parallel;
- "--batch": specifies how many trajectories are evaluated in each iteration;
- "--clip": specifies whether to apply action clipping, you can select "0" or "1";
- "--n_trials": specifies how many run of the same experiments has to be done.

Here is an example running PGPE on Swimmer:
```bash
python3 run.py --dir /your/path --alg pgpe --ite 100 --var 1 --pol linear --env swimmer --horizon 100 --gamma 1 --lr 0.1 --lr_strategy adam --n_workers 6 --clip 1 --batch 30 --n_trials 1
```


