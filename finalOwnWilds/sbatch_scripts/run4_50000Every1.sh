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

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 1 --target_eps 401 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 2 --target_eps 402 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 3 --target_eps 403 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 4 --target_eps 404 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 5 --target_eps 405 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 6 --target_eps 406 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 7 --target_eps 407 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 8 --target_eps 408 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 9 --target_eps 409 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 10 --target_eps 410 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 11 --target_eps 411 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 12 --target_eps 412 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 13 --target_eps 413 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 14 --target_eps 414 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 15 --target_eps 415 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 16 --target_eps 416 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 17 --target_eps 417 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 18 --target_eps 418 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 19 --target_eps 419 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 20 --target_eps 420 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 21 --target_eps 421 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 22 --target_eps 422 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 23 --target_eps 423 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 24 --target_eps 424 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 25 --target_eps 425 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 26 --target_eps 426 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 27 --target_eps 427 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 28 --target_eps 428 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 29 --target_eps 429 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 30 --target_eps 430 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run4 --save_step 200 --save_last False
