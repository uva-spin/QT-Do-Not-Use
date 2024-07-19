#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:4  
#SBATCH --cpus-per-task=4
#SBATCH -t 72:00:00
#SBATCH -J QTracking_Training_TFZ
#SBATCH -o Slurm_Files/QTracking_Training_TFZ.out
#SBATCH -e Slurm_Files/QTracking_TFZ.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Track_Finder_Training.py Z