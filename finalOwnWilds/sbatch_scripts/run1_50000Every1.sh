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

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 101 --target_eps 1 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 102 --target_eps 2 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 103 --target_eps 3 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 104 --target_eps 4 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 105 --target_eps 5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 106 --target_eps 6 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 107 --target_eps 7 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 108 --target_eps 8 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 109 --target_eps 9 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 110 --target_eps 10 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 111 --target_eps 11 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 112 --target_eps 12 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 113 --target_eps 13 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 114 --target_eps 14 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 115 --target_eps 15 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 116 --target_eps 16 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 117 --target_eps 17 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 118 --target_eps 18 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 119 --target_eps 19 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 120 --target_eps 20 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 121 --target_eps 21 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 122 --target_eps 22 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 123 --target_eps 23 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 124 --target_eps 24 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 125 --target_eps 25 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 126 --target_eps 26 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 127 --target_eps 27 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 128 --target_eps 28 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 129 --target_eps 29 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 130 --target_eps 30 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run1 --save_step 200 --save_last False
