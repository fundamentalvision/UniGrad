#!/usr/bin/env bash

CFG_PATH=$1
GPUS=$2
PORT=${PORT:-28500}

set -x

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
    pretrain.py $CFG_PATH ${@:3}