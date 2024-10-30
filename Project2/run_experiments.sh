#!/bin/sh
bsub -env "all, LD_LIBRARY_PATH=/work3/s204078/DeepVision/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH" < submit_gpu.sh
# bsub < submit_gpu.sh
