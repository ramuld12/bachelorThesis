#!/bin/bash
#BATCH --job-name=gridEpsSmallSet
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=10000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=1-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 1 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 2 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 3 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 4 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 5 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 6 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 7 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 8 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 9 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
python3 examples/run_expt.py --dataset civilcomments --algorithm groupDRO --root_dir data --batch_size 16 --n_epochs 20 --groupby_fields y --seed 10 --max_token_length 100 --log_dir ./logs/50000Baseline --save_step 20 --useDP False
