cd ..
# python run_phases.py \
#   --learn 0 \
#   --sigma_lr 0.01 \
#   --sigma_lr_strategy adam \
#   --phases 25 \
#   --sigma_exponent 1 \
#   --sigma_init 1 \
#   --dir /Users/ale/Desktop/results/sens/ \
#   --ite 40 \
#   --alg pgpe \
#   --var 1.0 \
#   --pol linear \
#   --env ip \
#   --horizon 100 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 5 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 

python run_phases.py \
  --learn 0 \
  --sigma_lr 0.01 \
  --sigma_lr_strategy adam \
  --phases 10 \
  --sigma_exponent 1 \
  --sigma_init 1 \
  --dir /Users/ale/Desktop/results/sens/ \
  --ite 100 \
  --alg pgpe \
  --var 1.0 \
  --pol linear \
  --env ip \
  --horizon 100 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 5 \
  --batch 100 \
  --clip 0 \
  --n_trials 5 

python run_phases.py \
  --learn 0 \
  --sigma_lr 0.01 \
  --sigma_lr_strategy adam \
  --phases 1000 \
  --sigma_exponent 1 \
  --sigma_init 1 \
  --dir /Users/ale/Desktop/results/sens/ \
  --ite 1 \
  --alg pgpe \
  --var 1.0 \
  --pol linear \
  --env ip \
  --horizon 100 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 5 \
  --batch 100 \
  --clip 0 \
  --n_trials 5

# python run_phases.py \
#   --learn 0 \
#   --sigma_lr 0.01 \
#   --sigma_lr_strategy adam \
#   --phases 25 \
#   --sigma_exponent 1 \
#   --sigma_init 1 \
#   --dir /Users/ale/Desktop/results/sens/ \
#   --ite 40 \
#   --alg pg \
#   --var 1.0 \
#   --pol linear \
#   --env ip \
#   --horizon 100 \
#   --gamma 1.0 \
#   --lr 0.01 \
#   --lr_strategy adam \
#   --n_workers 5 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 

python run_phases.py \
  --learn 0 \
  --sigma_lr 0.01 \
  --sigma_lr_strategy adam \
  --phases 10 \
  --sigma_exponent 1 \
  --sigma_init 1 \
  --sigma_param exp \
  --dir /Users/ale/Desktop/results/sens/ \
  --ite 100 \
  --alg pg \
  --var 1.0 \
  --pol linear \
  --env ip \
  --horizon 100 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 5 \
  --batch 100 \
  --clip 0 \
  --n_trials 5

python run_phases.py \
  --learn 0 \
  --sigma_lr 0.01 \
  --sigma_lr_strategy adam \
  --phases 1000 \
  --sigma_exponent 1 \
  --sigma_init 1 \
  --sigma_param exp \
  --dir /Users/ale/Desktop/results/sens/ \
  --ite 1 \
  --alg pg \
  --var 1.0 \
  --pol linear \
  --env ip \
  --horizon 100 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 5 \
  --batch 100 \
  --clip 0 \
  --n_trials 5
