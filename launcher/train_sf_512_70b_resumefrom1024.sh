#! /bin/bash
set -x
# This is the example script to pretrain a 7B LLaMA model on a TPU v4 pod. These
# hyperparameters are the ones we used to train the OpenLLaMA 7B model on
# the RedPajama dataset. To use this on TPU pod, you need to run this
# script on every hosts in a TPU pod.

# Put your WANDB API key here to enable logging to wandb.
export WANDB_API_KEY=""
export WANDB_BASE_URL="https://salesforceairesearch.wandb.io"
export WANDB_ENTITY="cxing"
export WANDB_PROJECT="lemur"

# TPU specific flags to improve training throughput
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
# export LIBTPU_INIT_ARGS=\"--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_enable_async_collective_offload=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE\"
export JAX_USE_PJRT_C_API_ON_TPU=1
export JAX_PLATFORMS=''
PROJECT_DIR=/root/lemur-tpu
OUTPUT_DIR=$PROJECT_DIR/outputs/open_llama_debug
mkdir -p $OUTPUT_DIR
echo "Number of Open File Limit before changing"
ulimit -n
ulimit -n 1048576
echo "Number of Open File Limit after changing"
ulimit -n

#eval "$(/home/gcpuser/miniconda3/condabin/conda shell.bash hook)" && conda activate xtpu
eval "$(/opt/miniconda/condabin/conda shell.bash hook)" && conda activate xtpu-newjax
#conda activate xtpu
cd $PROJECT_DIR && pip install -e .
process_id=$1
python -m xtpu.train \
    --mesh_dim='1,64,4' \
    --dtype='bf16' \
    --total_steps=630000 \
    --log_freq=128 \
    --save_model_freq=0 \
    --save_milestone_freq=10000 \
    --load_llama_config='llama-2-70b' \
    --update_llama_config='' \
    --load_dataset_state='gs://.../cxing/models/xtpu-dynamic-json-1024-70B-llama2-exp2/61ae7eb27beb4560ba55ee0bd2c2882d/dataset_10000_bs128_seqlen2048.pkl' \
    --load_checkpoint='trainstate_params::gs://.../cxing/models/xtpu-dynamic-json-1024-70B-llama2-exp2/61ae7eb27beb4560ba55ee0bd2c2882d/streaming_train_state_10000' \
    --tokenizer.vocab_file='gs://.../cxing/models/llama-2-70b-tpu/tokenizer.model' \
    --optimizer.type='adamw' \
    --optimizer.adamw_optimizer.weight_decay=0.1 \
    --optimizer.adamw_optimizer.lr=4e-5 \
    --optimizer.adamw_optimizer.end_lr=4e-6 \
    --optimizer.adamw_optimizer.lr_warmup_steps=8000 \
    --optimizer.adamw_optimizer.lr_decay_steps=630000 \
    --optimizer.accumulate_gradient_steps=16 \
    --train_dataset.type='multisource_json' \
    --train_dataset.text_processor.fields='text' \
    --train_dataset.multisource_json_dataset.path='gs://.../cxing/data/hk-data-v2/ds_info_v4.json' \
    --train_dataset.multisource_json_dataset.seq_length=2048 \
    --train_dataset.multisource_json_dataset.batch_size=128 \
    --train_dataset.multisource_json_dataset.tokenizer_processes=1 \
    --optimizer.adamw_optimizer.bf16_momentum=True \
    --checkpointer.save_optimizer_state=True \
    --logger.online=True \
    --logger.prefix='lemur' \
    --logger.project="lemur-tpu" \
    --logger.output_dir="gs://.../cxing/models/xtpu-dynamic-json-512-70B-llama2-exp7" \
    --logger.wandb_dir="$OUTPUT_DIR/wandb" \
    --jax_distributed.initialize_jax_distributed=True \
    --jax_distributed.coordinator_address='' \
    --jax_distributed.num_processes=64 \
    --jax_distributed.process_id=$process_id \
    > $HOME/xtpu-dynamic-json-512-70B-llama2-exp7.log 2>&1
