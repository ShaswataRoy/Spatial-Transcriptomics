#!/bin/sh -l

#SBATCH --mem=16g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS, 64 requests whole node

#SBATCH --partition=cpu   
#SBATCH --account=bczn-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=Spatial

#SBATCH --time=16:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -o detection.o          # Name of stdout output file
#SBATCH -e detection.e         # +Name of stderr error file

# Change to the directory from which you originally submitted this job.
cd $SLURM_SUBMIT_DIR

# module purge # Unload all loaded modules and reset everything to original state.
 
module load anaconda3_cpu

conda init bash
source activate spatial

python3 spot_detection.py
