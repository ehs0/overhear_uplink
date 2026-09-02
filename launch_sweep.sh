#!/usr/bin/env bash
# Launch the overhear / no-overhear rate-distortion sweep in parallel on CPU.
set -u
cd /home/ubuntu/overhear_uplink

run () {  # run <config> <output-dir> <log-name>
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 nohup .venv/bin/python -m overhear_uplink.train \
    --config "$1" --output-dir "$2" --device cpu --num-workers 2 \
    > "logs/$3.log" 2>&1 &
  echo "launched pid=$! config=$1 out=$2"
}

# Priority 1: no-overhear baseline at the same operating point as runs/synthetic
run configs/no_overhear.json            runs/no_overhear             no_overhear_l128
# Priority 4: remaining rate-distortion points (overhear l128 == runs/synthetic, already trained)
run configs/sweep/overhear_l32.json     runs/sweep/overhear_l32      overhear_l32
run configs/sweep/no_overhear_l32.json  runs/sweep/no_overhear_l32   no_overhear_l32
run configs/sweep/overhear_l64.json     runs/sweep/overhear_l64      overhear_l64
run configs/sweep/no_overhear_l64.json  runs/sweep/no_overhear_l64   no_overhear_l64
run configs/sweep/overhear_l256.json    runs/sweep/overhear_l256     overhear_l256
run configs/sweep/no_overhear_l256.json runs/sweep/no_overhear_l256  no_overhear_l256
