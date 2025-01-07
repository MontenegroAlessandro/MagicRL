cd ..
python run_phases.py \
  --phases 100 \
  --sigma_exponent 1.0 \
  --sigma_init 1.0 \
  --dir /Users/ale/Desktop/results \
  --ite 100 \
  --alg pgpe \
  --var 1.0 \
  --pol linear \
  --env swimmer \
  --costs 0 \
  --horizon 100 \
  --gamma 1.0 \
  --lr 0.01 \
  --lr_strategy adam \
  --n_workers 4 \
  --batch 100 \
  --clip 0 \
  --n_trials 1 
