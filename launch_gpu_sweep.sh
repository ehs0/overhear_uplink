#!/usr/bin/env bash
# Overhear / no-overhear rate-distortion sweep on GPU.
# All 8 operating points are trained from scratch under identical settings.
set -u
cd /home/ubuntu/overhear_uplink

run () {  # run <gpu> <config> <output-dir> <log-name>
  CUDA_VISIBLE_DEVICES="$1" PYTHONUNBUFFERED=1 nohup .venv/bin/python -m overhear_uplink.train \
    --config "$2" --output-dir "$3" --device cuda --num-workers 4 \
    > "logs/$4.log" 2>&1 &
  echo "launched pid=$! gpu=$1 config=$2 out=$3"
}

run 0 configs/sweep/overhear_l32.json     runs/sweep/overhear_l32     gpu_overhear_l32
run 0 configs/sweep/no_overhear_l32.json  runs/sweep/no_overhear_l32  gpu_no_overhear_l32
run 1 configs/sweep/overhear_l64.json     runs/sweep/overhear_l64     gpu_overhear_l64
run 1 configs/sweep/no_overhear_l64.json  runs/sweep/no_overhear_l64  gpu_no_overhear_l64
run 2 configs/sweep/overhear_l128.json    runs/sweep/overhear_l128    gpu_overhear_l128
run 2 configs/no_overhear.json            runs/no_overhear            gpu_no_overhear_l128
run 3 configs/sweep/overhear_l256.json    runs/sweep/overhear_l256    gpu_overhear_l256
run 3 configs/sweep/no_overhear_l256.json runs/sweep/no_overhear_l256 gpu_no_overhear_l256
