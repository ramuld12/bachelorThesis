#!/bin/bash
#BATCH --job-name=run4SmallEps
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=4-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4005 --target_eps 0.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4015 --target_eps 1.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4025 --target_eps 2.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4035 --target_eps 3.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4045 --target_eps 4.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4055 --target_eps 5.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4065 --target_eps 6.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4075 --target_eps 7.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4085 --target_eps 8.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4095 --target_eps 9.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4105 --target_eps 10.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4115 --target_eps 11.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4125 --target_eps 12.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4135 --target_eps 13.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4145 --target_eps 14.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4155 --target_eps 15.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4165 --target_eps 16.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4175 --target_eps 17.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4185 --target_eps 18.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4195 --target_eps 19.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4205 --target_eps 20.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4215 --target_eps 21.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4225 --target_eps 22.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4235 --target_eps 23.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4245 --target_eps 24.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4255 --target_eps 25.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4265 --target_eps 26.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4275 --target_eps 27.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4285 --target_eps 28.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4295 --target_eps 29.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run4 --save_step 200 --save_last False

