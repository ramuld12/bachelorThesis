#!/bin/bash
#BATCH --job-name=run1SmallEps
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=4-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3005 --target_eps 0.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3015 --target_eps 1.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3025 --target_eps 2.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3035 --target_eps 3.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3045 --target_eps 4.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3055 --target_eps 5.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3065 --target_eps 6.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3075 --target_eps 7.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3085 --target_eps 8.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3095 --target_eps 9.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3105 --target_eps 10.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3115 --target_eps 11.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3125 --target_eps 12.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3135 --target_eps 13.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3145 --target_eps 14.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3155 --target_eps 15.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3165 --target_eps 16.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3175 --target_eps 17.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3185 --target_eps 18.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3195 --target_eps 19.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3205 --target_eps 20.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3215 --target_eps 21.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3225 --target_eps 22.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3235 --target_eps 23.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3245 --target_eps 24.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3255 --target_eps 25.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3265 --target_eps 26.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3275 --target_eps 27.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3285 --target_eps 28.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3295 --target_eps 29.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run3 --save_step 200 --save_last False

