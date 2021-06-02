#!/bin/bash
#BATCH --job-name=run5SmallEps
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=4-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 501 --target_eps 1 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 502 --target_eps 2 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 503 --target_eps 3 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 504 --target_eps 4 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 505 --target_eps 5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 506 --target_eps 6 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 507 --target_eps 7 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 508 --target_eps 8 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 509 --target_eps 9 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 510 --target_eps 10 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 511 --target_eps 11 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 512 --target_eps 12 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 513 --target_eps 13 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 514 --target_eps 14 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 515 --target_eps 15 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 516 --target_eps 16 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 517 --target_eps 17 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 518 --target_eps 18 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 519 --target_eps 19 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 510 --target_eps 20 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 521 --target_eps 21 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 522 --target_eps 22 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 523 --target_eps 23 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 524 --target_eps 24 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 525 --target_eps 25 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 526 --target_eps 26 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 527 --target_eps 27 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 528 --target_eps 28 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 529 --target_eps 29 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 530 --target_eps 30 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every1Run5 --save_step 200 --save_last False
