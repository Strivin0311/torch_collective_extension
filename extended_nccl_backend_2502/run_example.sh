#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export WORLD_SIZE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

export MASTER_ADDRESS="localhost"
export MASTER_PORT=23457

export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

export PYTHONPATH=$PYTHONPATH:$(pwd)

export EXAMPLE_PROFILE_MODE=0
export EXAMPLE_SANITIZER_MODE=0
export EXAMPLE_USE_NCU_PROFILE_MODE=0


CMD="torchrun \
    --standalone \
    --nnode 1 \
    --nproc_per_node=$WORLD_SIZE \
    --master_addr=$MASTER_ADDRESS \
    --master_port=$MASTER_PORT \
    example.py
"

if [[ $EXAMPLE_PROFILE_MODE == "1" ]]; then
    if [[ $EXAMPLE_USE_NCU_PROFILE_MODE == "1" ]]; then
        ncu \
            --target-processes all \
            --set full \
            --kernel-name regex:".*group_reduce*" \
            -f -o example.ncu-rep \
            $CMD > example.log 2>&1
    else
        nsys profile \
            --force-overwrite true \
            -o example.nsys-rep \
            --capture-range=cudaProfilerApi \
            $CMD > example.log 2>&1
    fi
elif [[ $EXAMPLE_SANITIZER_MODE == "1" ]]; then
    compute-sanitizer --tool memcheck $CMD > example.log 2>&1
else
    $CMD > example.log 2>&1
fi
