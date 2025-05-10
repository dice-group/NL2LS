#!/bin/bash
set -eu

# To enable full paramter fine-tuning
export FULL_SFT=${1:-"false"}
# DeepSpeed configuration
export USE_DEEPSPEED==${2:-"true"}


# Determine the directory of the current script
CUR_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Change to current directory
cd $CUR_SCRIPT_DIR

PEFT_VENV_DIR=./venv-lola-custom-task
source $PEFT_VENV_DIR/bin/activate

## Specify CUDA device
#export CUDA_VISIBLE_DEVICES="0"
#export CUDA_VISIBLE_DEVICES="0,1"

# CUDA configuration
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
else
    GPU_COUNT=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
fi
echo "Assigning ${GPU_COUNT} gpu(s) for training"

# Debug mode and prefix setup
postfix=$(date +"%d%m%y%H%M%S")

LAUNCHER="python"
prefix=""

EXTRA_PARAMS=()

if [[ "$FULL_SFT" == true ]]; then
    prefix="full_sft-"
    EXTRA_PARAMS+=("--full_sft")
fi

if [[ "$USE_DEEPSPEED" == true ]]; then
    #prefix="deepspeed-"
    EXTRA_PARAMS+=("--deepspeed" "deepspeed_config.json")
fi

# Model and task configuration
MODEL_HF_ID=dice-research/lola_v1
MODEL_NAME="${MODEL_HF_ID##*/}"

DATA_ROOT=data_dir
LANG=de
DATASET_NAME=limes-silver

DATA_DIR=${DATA_ROOT}/${LANG}/${DATASET_NAME}

RUN_LABEL=${prefix}${MODEL_NAME}-${LANG}-${DATASET_NAME}

# Training command with extensible parameters
$LAUNCHER -m torch.distributed.run --nnodes=1 --nproc_per_node=$GPU_COUNT --master_port=4550 train_custom_task.py \
    --model_name_or_path $MODEL_HF_ID \
    --do_train \
    --train_data_path $DATA_DIR/train.json \
    --output_dir trained_model/$RUN_LABEL\
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --logging_steps 10 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --run_name $RUN_LABEL \
    "${EXTRA_PARAMS[@]}"