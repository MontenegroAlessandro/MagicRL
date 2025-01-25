cd ..

taskset -ca 0-35 python run_phases.py \
  --learn 0 \
  --sigma_lr 0.1 \
  --sigma_lr_strategy adam \
  --phases 25 \
  --sigma_exponent 1.0 \
  --sigma_init 1.0 \
  --dir /data2/alessandro/apes/ \
  --ite 200 \
  --alg pg \
  --var 1.0 \
  --pol linear \
  --env swimmer \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 36 \
  --batch 100 \
  --clip 0 \
  --n_trials 5

taskset -ca 0-35 python run_phases.py \
  --learn 1 \
  --sigma_lr 0.1 \
  --sigma_lr_strategy adam \
  --phases 25 \
  --sigma_exponent 1.0 \
  --sigma_init 1.0 \
  --dir /data2/alessandro/apes/ \
  --ite 200 \
  --alg pg \
  --var 1.0 \
  --pol linear \
  --env swimmer \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 36 \
  --batch 100 \
  --clip 0 \
  --n_trials 5

taskset -ca 0-35 python run_phases.py \
  --learn 0 \
  --sigma_lr 0.1 \
  --sigma_lr_strategy adam \
  --phases 25 \
  --sigma_exponent 1.0 \
  --sigma_init 1.0 \
  --dir /data2/alessandro/pes/ \
  --ite 300 \
  --alg pgpe \
  --var 1.0 \
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

taskset -ca 0-35 python run_phases.py \
  --learn 1 \
  --sigma_lr 0.1 \
  --sigma_lr_strategy adam \
  --phases 25 \
  --sigma_exponent 1.0 \
  --sigma_init 1.0 \
  --dir /data2/alessandro/pes/ \
  --ite 300 \
  --alg pgpe \
  --var 1.0 \
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
