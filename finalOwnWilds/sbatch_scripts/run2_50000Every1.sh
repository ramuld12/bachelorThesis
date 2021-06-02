#!/bin/bash
#BATCH --job-name=run2SmallEps
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=4-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 201 --target_eps 1 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 202 --target_eps 2 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 203 --target_eps 3 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 204 --target_eps 4 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 205 --target_eps 5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 206 --target_eps 6 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 207 --target_eps 7 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 208 --target_eps 8 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 209 --target_eps 9 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 210 --target_eps 10 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 211 --target_eps 11 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 212 --target_eps 12 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 213 --target_eps 13 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 214 --target_eps 14 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 215 --target_eps 15 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 216 --target_eps 16 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 217 --target_eps 17 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 218 --target_eps 18 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 219 --target_eps 19 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 210 --target_eps 20 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 221 --target_eps 21 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 222 --target_eps 22 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 223 --target_eps 23 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 224 --target_eps 24 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 225 --target_eps 25 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 226 --target_eps 26 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 227 --target_eps 27 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 228 --target_eps 28 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 229 --target_eps 29 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 230 --target_eps 30 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run2 --save_step 200 --save_last False
