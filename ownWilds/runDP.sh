#!/bin/bash
#SBATCH --job-name=trainCivil
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=8000M
#SBATCH -p gpu --gres=gpu:titanx:1
#SBATCH --time=2-00:00:00

echo "Training started by szl855"
echo $CUDA_VISIBLE_DEVICES
source scl_source enable devtoolset-7

python3 -m pip install --user torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

python3 -c "import torch; print(torch.cuda.is_available())"
python3 -c "import torch; print(torch.__version__)"
python3 -c "import torch; print(torch.version.cuda)"

python3 -m pip install --user torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
python3 -m pip install --user torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
python3 -m pip install --user torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
python3 -m pip install --user torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.1+cu110.html
python3 -m pip install --user torch-geometric
python3 -m pip install --user transformers

python3 -m pip install --user wilds 
python3 -c "import wilds; print(wilds.__version__)"

python3 -m pip install scipy
python3 -m pip install mpmath
python3 -m pip install tensorflow
python3 -m pip install tensorflow-privacy

python3 examples/run_expt.py --dataset civilcomments --download --algorithm groupDRO --root_dir data --progress_bar --optimizer PrivateAdam
