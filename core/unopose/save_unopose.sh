#!/usr/bin/env bash
# test
set -x
this_dir=$(dirname "$0")
# commonly used opts:

# misc.load_from: resume or pretrained, or test checkpoint
CFG=$1
CUDA_VISIBLE_DEVICES=$2
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
# GPUS=($(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n'))
NGPU=${#GPUS[@]}  # echo "${GPUS[0]}"
echo "use gpu ids: $CUDA_VISIBLE_DEVICES num gpus: $NGPU"
CKPT=$3
if [ ! -e "$CKPT" ]; then
    echo "$CKPT does not exist."
    exit 1
fi
NCCL_DEBUG=INFO
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONPATH="$this_dir/../..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=$2 python $this_dir/main_unopose.py \
    --config-file $CFG --num-gpus $NGPU \
    test.save_results_only=True misc.load_from=$CKPT \
    ${@:4}
