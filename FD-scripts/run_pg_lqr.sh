#!/bin/bash

# PG with gaussian and linear policies on LQR
# for state_action_dim in 2 4;
# do
#   for pol in "nn";
#   do
#     for var in 0.1 0.01 0.001 0.0001;
#     do
#       for lr in 0.01 0.001 0.0001;
#       do
#         python ./run.py \
#         --dir ./results_lqr_pg/ \
#         --ite 1500 \
#         --alg pg \
#         --var $var \
#         --pol $pol \
#         --env lqr \
#         --horizon 50 \
#         --gamma 1.0 \
#         --lr $lr \
#         --lr_strategy adam \
#         --n_workers 6 \
#         --batch 100 \
#         --clip 0 \
#         --n_trials 3 \
#         --lqr_action_dim $state_action_dim \
#         --lqr_state_dim $state_action_dim
#       done
#     done
#   done
# done

# PG-FD with linear and nn policies on LQR 2x2
# stochastic/trajectory/central/linear/_04_24-21_39_54_seed_0_pg_fd_750_lqr_50_adam_001_linear_batch_100_clip_4_var_01
for state_action_dim in 2;
do
  for pol in linear nn;
  do
    for lr in 0.01 0.001 0.0001;
    do
      for var in 0.1 0.01 0.001 0.0001;
      do
        for rollout_mode in stochastic;
        do
          for perturb_scope in step trajectory;
          do
            for fd_mode in forward central;
            do
              python ./run.py \
              --dir ./results_lqr_pg_fd_new/$rollout_mode/$perturb_scope/$fd_mode/$pol/ \
              --ite 750 \
              --alg pg_fd \
              --var $var \
              --pol $pol \
              --env lqr \
              --horizon 50 \
              --gamma 1.0 \
              --lr $lr \
              --lr_strategy adam \
              --n_workers 6 \
              --batch 100 \
              --clip 1 \
              --n_trials 3 \
              --lqr_action_dim $state_action_dim \
              --lqr_state_dim $state_action_dim \
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
