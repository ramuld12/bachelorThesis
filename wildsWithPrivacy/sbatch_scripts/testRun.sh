#!/bin/bash
#SBATCH --job-name=trainCivil
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=8000M
#SBATCH --time=2-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES
source scl_source enable devtoolset-7

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"
python3 -c "import wilds; print(wilds.__version__)"

python3 -m pip install opacus

python3 examples/run_expt.py --dataset civilcomments --download --algorithm groupDRO --root_dir data --progress_bar --optimizer DPSGD
