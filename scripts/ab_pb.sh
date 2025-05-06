cd ..
python3 run.py \
    --dir /Users/ale/Desktop/results/ \
    --ite 5000 \
    --alg pgpe \
    --var 0.5 \
    --pol linear \
    --env reacher \
    --horizon 50 \
    --gamma 1 \
    --lr 0.001 \
    --lr_strategy adam \
    --n_workers 5 \
    --batch 100 \
    --n_trials 1 \
    --clip 0