#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export MASTER_ADDRESS="localhost"
export MASTER_PORT=23457

export OMP_NUM_THREADS=1

torchrun \
    --standalone \
    --nnode 1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDRESS \
    --master_port=$MASTER_PORT \
    example.py > example.log 2>&1