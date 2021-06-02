#!/bin/bash
#BATCH --job-name=gridEpsSmallSet
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time=4-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 1 --target_eps 1 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 5 --target_eps 5 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 10 --target_eps 10 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 20 --target_eps 20 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 30 --target_eps 30 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 40 --target_eps 40 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 50 --target_eps 50 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 60 --target_eps 60 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 70 --target_eps 70 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 80 --target_eps 80 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 90 --target_eps 90 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every5 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 100 --target_eps 100 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 200 --target_eps 200 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 300 --target_eps 300 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 400 --target_eps 400 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 500 --target_eps 500 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 600 --target_eps 600 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 700 --target_eps 700 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 800 --target_eps 800 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 900 --target_eps 900 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 8 --n_epochs 20 --groupby_fields y --seed 1000 --target_eps 1000 --max_token_length 100 --target_delta 1e-05 --log_dir ./logs/50000Every100 --save_step 20
