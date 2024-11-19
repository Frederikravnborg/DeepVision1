#!/bin/sh
#BSUB -J z
#BSUB -e z%J.err
#BSUB -q c02516
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4,16
#BSUB -u
#BSUB -B
#BSUB -R "rusage[mem=35G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 12:00
# load a scipy module
# replace VERSION and uncomment
# module load cuda/12.2.2

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!

python3 faster_rcnn_train2.py

