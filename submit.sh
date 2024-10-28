#!/bin/sh
#BSUB -J job
#BSUB -o job_%J.out
#BSUB -e job_%J.err
#BSUB -q hpc
#BSUB -n 1
#BSUB -u 
#BSUB -R "rusage[mem=35G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10
# end of BSUB options

# load a scipy module
# replace VERSION and uncomment
module load cuda/12.2.2

# activate the virtual environment
# NOTE: needs to have been built with the same SciPy version above!
#source deepvision/bin/activate

python3 part1_main.py
