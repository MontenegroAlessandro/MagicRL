@echo off
echo Running experiment 1...
python run.py --alg off_pg --pol "nn" --lr 0.0001 --env half_cheetah --n_workers 1 --var 0.1 --horizon 100 --ite 2500 --window_length 8 --weight_type MIS --n_trials 10 --clip 0 --batch 40 --dir results\MIS\

echo Running experiment 2...
python run.py --alg off_pg --env cart_pole --lr 0.01 --n_workers 10 --var 0.1 --horizon 100 --ite 1000 --window_length 8 --weight_type MIS --n_trials 10 --clip 0 --batch 10 --dir results\MIS\

echo Running experiment 3...
python run.py --alg off_pg --pol "nn" --lr 0.001 --env swimmer --n_workers 10 --var 0.1 --horizon 100 --ite 5000 --window_length 4 --weight_type MIS --n_trials 5 --clip 0 --batch 20 --dir results\MIS\

echo All experiments completed!
pause