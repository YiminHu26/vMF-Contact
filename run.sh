#!/bin/bash
#SBATCH --job-name=example_job           # Job name
#SBATCH --output=output_%j.log           # Output log file (%j expands to jobID)
#SBATCH --error=error_%j.log             # Error log file
#SBATCH --mem=100G                        # Total memory per task
#SBATCH --time=01:00:00                  # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:1                     # Number of GPUs (if needed)
#SBATCH --partition=accelerated          # Partition to submit to
#SBATCH --account=hk-project-test-p0023465

# Load any necessary modules
conda activate c3d

module load devel/cuda/12.2

# Change to the directory where the job was submitted from
source ~/.bashrc

# Print some information about the job
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated CPUs: $SLURM_CPUS_ON_NODE"
echo "Allocated GPUs: $SLURM_JOB_GPUS"

# Execute your script or command
python vmf_contact/train.py