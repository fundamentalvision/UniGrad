#!/usr/bin/env bash

PRETRAINED_PATH=$1
DATA_PATH=$2
GPUS=$3
PORT=${PORT:-28500}

set -x

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT --use_env \
    lineval.py --pretrained $PRETRAINED_PATH --data $DATA_PATH --lars ${@:4}