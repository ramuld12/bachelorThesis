#!/bin/bash
#BATCH --job-name=run3SmallEps
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=4-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 301 --target_eps 1 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 302 --target_eps 2 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 303 --target_eps 3 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 304 --target_eps 4 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 305 --target_eps 5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 306 --target_eps 6 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 307 --target_eps 7 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 308 --target_eps 8 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 309 --target_eps 9 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 310 --target_eps 10 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 311 --target_eps 11 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 312 --target_eps 12 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 313 --target_eps 13 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 314 --target_eps 14 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 315 --target_eps 15 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 316 --target_eps 16 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 317 --target_eps 17 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 318 --target_eps 18 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 319 --target_eps 19 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 310 --target_eps 20 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 321 --target_eps 21 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 322 --target_eps 22 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 323 --target_eps 23 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 324 --target_eps 24 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 325 --target_eps 25 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 326 --target_eps 26 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 327 --target_eps 27 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 328 --target_eps 28 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 329 --target_eps 29 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 330 --target_eps 30 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run3 --save_step 200 --save_last False
