#!/bin/bash
#SBATCH -A spinquest
#SBATCH -p gpu
#SBATCH --gres=gpu:4  
#SBATCH --cpus-per-task=4
#SBATCH -t 12:00:00
#SBATCH -J QTracking_Event_Filter
#SBATCH -o Slurm_Files/QTracking_Event_Filter.out
#SBATCH -e Slurm_Files/QTracking_Event_Filter.err
#SBATCH --mem=256000

module purge
module load apptainer tensorflow/2.13.0

apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Make_Networks.py
apptainer run --nv $CONTAINERDIR/tensorflow-2.13.0.sif Python_Files/Event_Filter_Training.py
