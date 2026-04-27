# cd ..
# taskset -ca 0-11 python run.py \
#   --dir /data2/alessandro/pgpe/ \
#   --ite 5000 \
#   --alg pgpe \
#   --var 0.0016 \
#   --pol linear \
#   --env swimmer \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 12 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5

# taskset -ca 0-11 python run.py \
#   --dir /data2/alessandro/pg/ \
#   --ite 5000 \
#   --alg pg \
#   --var 1 \
#   --pol linear \
#   --env swimmer \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 12 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 

# taskset -ca 0-11 python run.py \
#   --dir /data2/alessandro/pg/ \
#   --ite 5000 \
#   --alg pg \
#   --var 0.25 \
#   --pol linear \
#   --env swimmer \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 12 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 

#   taskset -ca 0-11 python run.py \
#   --dir /data2/alessandro/pg/ \
#   --ite 5000 \
#   --alg pg \
#   --var 0.0016 \
#   --pol linear \
#   --env swimmer \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 12 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 


# for lr in 0.1 0.3 0.5; 
#   do
#   for var in 0.1 0.01 0.05 0.001;
#    do
#     python run.py \
#     --dir ./results_pgpefd/ \
#     --ite 250 \
#     --alg pgpe_fd \
#     --var $var \
#     --pol linear \
#     --env swimmer \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr $lr \
#     --lr_strategy adam \
#     --n_workers 6 \
#     --batch 100 \
#     --clip 0 \
#     --n_trials 3
#   done
# done

# for state_action_dim in 2 4 8;
#   do
#   for pol in "nn" "linear";
#   do
#     for var in 0.1 0.05 0.01 0.001;
#     do
#       python run.py \
#       --dir ./test_lqr_latest/ \
#       --ite 500 \
#       --alg pgpe \
#       --var $var \
#       --pol $pol \
#       --env lqr \
#       --horizon 50 \
#       --gamma 1.0 \
#       --lr 0.01 \
#       --lr_strategy adam \
#       --n_workers 6 \
#       --batch 100 \
#       --clip 0 \
#       --n_trials 3 \
#       --lqr_action_dim $state_action_dim\
#       --lqr_state_dim $state_action_dim
#     done
#   done
#   done

for state_action_dim in 2 4 8;
  do
  for pol in "nn" "linear";
    do
    for mode in forward central five_point;
    do
      for var in 0.1 0.05 0.01 0.001 0.0001 0.00001;
      do
        python run.py \
        --dir ./test_lqr_latest_pgpe-fd/ \
        --ite 150 \
        --alg pgpe_fd \
        --var $var \
        --pol $pol \
        --env lqr \
        --horizon 50 \
        --gamma 1.0 \
        --lr 0.01 \
        --lr_strategy adam \
        --n_workers 6 \
        --batch 100 \
        --clip 0 \
        --n_trials 3 \
        --fd_mode $mode \
        --lqr_action_dim $state_action_dim\
        --lqr_state_dim $state_action_dim
      done
    done
  done
done
# for var in 0.5 0.1 0.05 0.01 0.001 0.0005 0.0001 0.00005 0.00001;
#   do
#   python run.py \
#   --dir ./test_lqr_2/ \
#   --ite 1500 \
#   --alg pgpe \
#   --var $var \
#   --pol linear \
#   --env lqr \
#   --horizon 50 \
#   --gamma 1.0 \
#   --lr 0.1 \
#   --lr_strategy adam \
#   --n_workers 6 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 3
# done


  #   python run.py \
  # --dir ./test_swimmer/ \
  # --ite 4250 \
  # --alg pgpe \
  # --var 0.3 \
  # --pol linear \
  # --env swimmer \
  # --horizon 200 \
  # --gamma 1.0 \
  # --lr 0.1 \
  # --lr_strategy adam \
  # --n_workers 6 \
  # --batch 100 \
  # --clip 1 \
  # --n_trials 3

# for var in 0.1;
#   do
#   for lr in 0.5;
#     do
#       python run.py \
#     --dir ./test_swimmer/ \
#     --ite 250 \
#     --alg pgpe_fd \
#     --var $var \
#     --pol linear \
#     --env swimmer \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr $lr \
#     --lr_strategy adam \
#     --n_workers 6 \
#     --batch 100 \
#     --clip 1 \
#     --n_trials 3
#     done
#   done


  #   python run.py \
  # --dir ./test_pendulum/ \
  # --ite 4250 \
  # --alg pgpe \
  # --var 0.1 \
  # --pol linear \
  # --env ip \
  # --horizon 200 \
  # --gamma 1.0 \
  # --lr 0.1 \
  # --lr_strategy adam \
  # --n_workers 6 \
  # --batch 100 \
  # --clip 0 \
  # --n_trials 3


# for var in 0.1 0.01 0.001 0.0001;
#   do
#   for lr in 0.5 0.1 0.01;
#     do
#       python run.py \
#     --dir ./test_pendulum/ \
#     --ite 250 \
#     --alg pgpe_fd \
#     --var $var \
#     --pol linear \
#     --env ip \
#     --horizon 200 \
#     --gamma 1.0 \
#     --lr $lr \
#     --lr_strategy adam \
#     --n_workers 6 \
#     --batch 100 \
#     --clip 0 \
#     --n_trials 3
#     done
#   done

  #     python run.py \
  # --dir ./test_pgpe/ \
  # --ite 250 \
  # --alg pg \
  # --var 0.3 \
  # --pol linear \
  # --env swimmer \
  # --horizon 200 \
  # --gamma 1.0 \
  # --lr 0.01 \
  # --lr_strategy adam \
  # --n_workers 6 \
  # --batch 100 \
  # --clip 0 \
  # --n_trials 5

  #     python run.py \
  # --dir ./test_pgpe/ \
  # --ite 250 \
  # --alg pg_fd \
  # --var 0.3 \
  # --pol linear \
  # --env swimmer \
  # --horizon 200 \
  # --gamma 1.0 \
  # --lr 0.01 \
  # --lr_strategy adam \
  # --n_workers 6 \
  # --batch 100 \
  # --clip 0 \
  # --n_trials 5