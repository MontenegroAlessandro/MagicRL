#!/bin/bash

# PG with NN policies on MuJoCo environments (Swimmer, Hopper, HalfCheetah)
# Usage: bash run_pg_mujoco.sh

for env in hopper;
do
  # Usually, linear policies struggle on complex MuJoCo envs, so we restrict to 'nn'
  for pol in "linear" "nn";
  do
    for var in 0.1 0.01;
    do
      for lr in 0.1 0.01;
      do
        echo "Running PG | Env: $env | Pol: $pol | Var: $var | LR: $lr"
        
        python ./run.py \
        --dir ./results_mujoco_pg/$env/ \
        --ite 500 \
        --alg pg \
        --var $var \
        --pol $pol \
        --env $env \
        --horizon 200 \
        --gamma 1 \
        --lr $lr \
        --lr_strategy adam \
        --n_workers 4 \
        --batch 100 \
        --clip 0 \
        --n_trials 3
      done
    done
  done
done



for env in hopper;
do
  # You can remove 'linear' if you only want to test neural networks on MuJoCo
  for pol in linear nn; 
  do
    for lr in 0.01 0.001 0.0001;
    do
      for var in 0.1 0.01 0.001 0.0001;
      do
        for rollout_mode in deterministic stochastic;
        do
          for perturb_scope in step trajectory;
          do
            for fd_mode in forward central;
            do
              echo "Running PG-FD | Env: $env | Pol: $pol | Scope: $perturb_scope | FD: $fd_mode | LR: $lr"
              
              # Constructing a dynamic directory path similar to your original script
              python ./run.py \
              --dir ./results_mujoco_pg_fd/$env/$rollout_mode/$perturb_scope/$fd_mode/$pol/ \
              --ite 250 \
              --alg pg_fd \
              --var $var \
              --pol $pol \
              --env $env \
              --horizon 500 \
              --gamma 1 \
              --lr $lr \
              --lr_strategy adam \
              --n_workers 4 \
              --batch 100 \
              --clip 1 \
              --n_trials 3 \
              --pg_fd_rollout_mode $rollout_mode \
              --pg_fd_perturbation_scope $perturb_scope \
              --fd_mode $fd_mode
            done
          done
        done
      done
    done
  done
done