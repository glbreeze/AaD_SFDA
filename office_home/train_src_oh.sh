#!/bin/bash

#SBATCH --job-name=LP_BN
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu
#SBATCH --partition=a100_1,a100_2,v100,rtx8000

# job info

# Singularity path
ext3_path=/scratch/$USER/python36/python36.ext3
sif_path=/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
source /ext3/env.sh
python train_src.py --dset p2c --home

python train_src.py --dset p2r --home

python train_src.py --dset p2a --home

python train_src.py --dset a2p --home

python train_src.py --dset a2r --home

python train_src.py --dset a2c --home

python train_src.py --dset r2a --home

python train_src.py --dset r2p --home

python train_src.py --dset r2c --home

python train_src.py --dset c2r --home

python train_src.py --dset c2a --home

python train_src.py --dset c2p --home
"




