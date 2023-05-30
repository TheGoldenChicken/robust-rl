#!/bin/sh
#BSUB -J SUMO
#BSUB -o SUMO_%J.out
#BSUB -e SUMO_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options


# load CUDA (for GPU support)

# activate the virtual environment
source $HOME/miniconda3/envs/sumo/bin/activate

python sumo_experiments.py
