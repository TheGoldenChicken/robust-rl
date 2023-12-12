#!/bin/sh 


### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err

### General options 
### -- specify queue -- 
#BSUB -q gpuv100
#BSUB -gpu "num=1"

### -- set the job Name -- 
#BSUB -J dqn_rl_network

### -- ask for 1 core -- 
#BSUB -n 8

### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=3GB]"
#BSUB -R "span[hosts=1]"

### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB

### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 

# load CUDA (for GPU support)
module load cuda/11.3
module load python3/3.9.11

python3 -m venv venv_1

source venv_1/bin/activate

pip install --upgrade urllib3==1.26.15

python3 -m pip install -r requirements.txt

# python3 -m install --upgrade urllib3
# python3 -m uninstall urllib3
# python3 -m install 'urllib3<2.0'

python experiment.py --wandb_key ec26ff6ba9b98d017cdb3165454ce21496c12c35 \
       --test_interval 1000 --train_frames 50000 \
       --robust_batch_size 128 \
       --grad_batch_size 128 \
       --learning_rate 0.00005 \
       --radial_basis_dist 2 \
       --radial_basis_var 3 \
       --delta 0 --seed 1 2 3\
       --non_linear \
       --noise_var 0.01 \
       --gamma 0.99 \
       --bin_size 100000 \
       --train_identifier dqn_RBF
