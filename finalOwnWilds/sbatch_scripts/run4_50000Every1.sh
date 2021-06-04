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

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 401 --target_eps 01 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 402 --target_eps 02 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 403 --target_eps 03 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 404 --target_eps 04 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 405 --target_eps 05 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 406 --target_eps 06 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 407 --target_eps 07 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 408 --target_eps 08 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 409 --target_eps 09 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 410 --target_eps 10 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 411 --target_eps 11 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 412 --target_eps 12 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 413 --target_eps 13 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 414 --target_eps 14 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 415 --target_eps 15 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 416 --target_eps 16 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 417 --target_eps 17 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 418 --target_eps 18 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 419 --target_eps 19 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 420 --target_eps 20 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 421 --target_eps 21 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 422 --target_eps 22 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 423 --target_eps 23 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 424 --target_eps 24 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 425 --target_eps 25 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 426 --target_eps 26 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 427 --target_eps 27 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 428 --target_eps 28 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 429 --target_eps 29 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 430 --target_eps 30 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
