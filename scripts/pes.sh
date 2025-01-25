cd ..
# python run_phases.py \
#   --learn 1 \
#   --sigma_lr 0.1 \
#   --sigma_lr_strategy adam \
#   --phases 25 \
#   --sigma_exponent 5.0 \
#   --sigma_init 0.5 \
#   --dir /Users/ale/Desktop/results/ \
#   --ite 200 \
#   --alg pg \
#   --var 1.0 \
#   --pol linear \
#   --env reacher \
#   --horizon 50 \
#   --gamma 1.0 \
#   --lr 0.001 \
#   --lr_strategy adam \
#   --n_workers 5 \
#   --batch 100 \
#   --clip 0 \
#   --n_trials 5 

python run_phases.py \
  --learn 1 \
  --sigma_lr 0.1 \
  --sigma_lr_strategy adam \
  --phases 1250 \
  --sigma_exponent 1 \
  --sigma_init 1 \
  --sigma_param exp \
  --dir /Users/ale/Desktop/results/ \
  --ite 1 \
  --alg pg \
  --var 1.0 \
  --pol linear \
  --env ip \
  --horizon 200 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 5 \
  --batch 100 \
  --clip 0 \
  --n_trials 5 
