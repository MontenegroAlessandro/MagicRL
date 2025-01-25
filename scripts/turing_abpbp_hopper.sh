cd ..

taskset -ca 0-35 python run.py \
  --dir /data1/montenegro/pgpe/ \
  --ite 7500 \
  --alg pgpe \
  --var 0.01 \
  --pol linear \
  --env hopper \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 36 \
  --batch 100 \
  --clip 0 \
  --n_trials 5

taskset -ca 0-35 python run.py \
  --dir /data1/montenegro/pgpe/ \
  --ite 7500 \
  --alg pgpe \
  --var 0.049 \
  --pol linear \
  --env hopper \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 36 \
  --batch 100 \
  --clip 0 \
  --n_trials 5

# taskset -ca 12-35 python run.py \
#   --dir /data2/alessandro/pgpe/ \
#   --ite 7500 \
#   --alg pgpe \
#   --var 1 \
#   --pol linear \
#   --env hopper \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 24 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5

#   taskset -ca 12-35 python run.py \
#   --dir /data2/alessandro/pgpe/ \
#   --ite 7500 \
#   --alg pgpe \
#   --var 0.25 \
#   --pol linear \
#   --env hopper \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 24 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5

# taskset -ca 12-35 python run.py \
#   --dir /data2/alessandro/pgpe/ \
#   --ite 7500 \
#   --alg pgpe \
#   --var 0.0016 \
#   --pol linear \
#   --env hopper \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 24 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5

# taskset -ca 12-35 python run.py \
#   --dir /data2/alessandro/pg/ \
#   --ite 7500 \
#   --alg pg \
#   --var 1 \
#   --pol linear \
#   --env hopper \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 24 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 

# taskset -ca 12-35 python run.py \
#   --dir /data2/alessandro/pg/ \
#   --ite 7500 \
#   --alg pg \
#   --var 0.25 \
#   --pol linear \
#   --env hopper \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 24 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 

#   taskset -ca 12-35 python run.py \
#   --dir /data2/alessandro/pg/ \
#   --ite 7500 \
#   --alg pg \
#   --var 0.0016 \
#   --pol linear \
#   --env hopper \
#   --horizon 200 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 24 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 