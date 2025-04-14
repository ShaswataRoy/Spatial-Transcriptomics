#!/bin/sh -l

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16    # <- match to OMP_NUM_THREADS, 64 requests whole node
#SBATCH --partition=gpuA100x4    # <- one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bczn-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=Spatial
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=05:00:00        # Total run time limit (hh:mm:ss)
#SBATCH --gpu-bind=verbose,per_task:1            # Queue (partition) name
#SBATCH -o inference.o          # Name of stdout output file
#SBATCH -e inference.e         # +Name of stderr error file

# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR

# module purge # Unload all loaded modules and reset everything to original state.
# module load python

module load anaconda3_gpu

conda init bash
source activate spatial


python3 create_mask.py

# sinteractive -A cis220051-gpu --time=02:00:00 --gpus-per-node=1 -p gpu 