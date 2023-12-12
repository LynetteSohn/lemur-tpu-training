set -x
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python-is-python3 \
    tmux \
    htop \
    git \
    nodejs \
    bmon \
    p7zip-full \
    nfs-common \
    wget

if [ ! -d "/opt/miniconda" ]; then
wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    eval "$(/opt/miniconda/bin/conda shell.bash hook)" && conda init && conda config --set auto_activate_base true && \
    rm /tmp/miniconda.sh
fi

eval "$(/opt/miniconda/bin/conda shell.bash hook)" && source ~/.bashrc
conda activate xtpu-0818
if [ $? -ne 0 ]; then
    conda create -n xtpu-0818 python=3.9 -y
    conda activate xtpu-0818
fi

# Python dependencies
cat > /tmp/tpu_requirements.txt <<- EndOfFile
-f https://storage.googleapis.com/jax-releases/libtpu_releases.html
jax[tpu]==0.4.13
flax==0.7.0
optax==0.1.7
distrax==0.1.3
chex==0.1.7
einops
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.0.1
transformers==4.31.0
datasets==2.14.2
huggingface_hub==0.16.4
tqdm
h5py
ml_collections
wandb==0.13.5
gcsfs==2022.11.0
requests
typing-extensions
git+https://github.com/EleutherAI/lm-evaluation-harness.git@big-refactor
mlxu==0.1.11
sentencepiece
EndOfFile

python -m pip install --upgrade -r /tmp/tpu_requirements.txt