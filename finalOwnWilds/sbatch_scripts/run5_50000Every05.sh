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

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2005 --target_eps 0.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2015 --target_eps 1.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2025 --target_eps 2.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2035 --target_eps 3.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2045 --target_eps 4.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2055 --target_eps 5.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2065 --target_eps 6.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2075 --target_eps 7.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2085 --target_eps 8.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2095 --target_eps 9.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2105 --target_eps 10.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2115 --target_eps 11.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2125 --target_eps 12.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2135 --target_eps 13.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2145 --target_eps 14.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2155 --target_eps 15.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2165 --target_eps 16.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2175 --target_eps 17.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2185 --target_eps 18.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2195 --target_eps 19.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2205 --target_eps 20.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2215 --target_eps 21.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2225 --target_eps 22.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2235 --target_eps 23.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2245 --target_eps 24.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2255 --target_eps 25.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2265 --target_eps 26.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2275 --target_eps 27.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2285 --target_eps 28.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2295 --target_eps 29.5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every05Run2 --save_step 200 --save_last False

