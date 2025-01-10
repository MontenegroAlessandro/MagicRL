cd ..

taskset -ca 24-35 python run.py \
  --dir /data2/alessandro/pg/ \
  --ite 5000 \
  --alg pg \
  --var 1 \
  --pol linear \
  --env swimmer \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.001 \
  --lr_strategy adam \
  --n_workers 12 \
  --batch 100 \
  --clip 0 \
  --n_trials 5 

taskset -ca 24-35 python run.py \
  --dir /data2/alessandro/pg/ \
  --ite 5000 \
  --alg pg \
  --var 0.25 \
  --pol linear \
  --env swimmer \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.001 \
  --lr_strategy adam \
  --n_workers 12 \
  --batch 100 \
  --clip 0 \
  --n_trials 5 

  taskset -ca 24-35 python run.py \
  --dir /data2/alessandro/pg/ \
  --ite 5000 \
  --alg pg \
  --var 0.0016 \
  --pol linear \
  --env swimmer \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.001 \
  --lr_strategy adam \
  --n_workers 12 \
  --batch 100 \
  --clip 0 \
  --n_trials 5 