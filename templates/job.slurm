#!/bin/bash -l
# Edit the account, partition, modules, and example resource requests for your HPC.
#SBATCH --job-name=__JOB_NAME__
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -e
module load YOUR_CONDA_MODULE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate meld_new
module load YOUR_CUDA_MODULE

if ! command -v nvcc >/dev/null 2>&1; then
    printf '%s\n' 'nvcc is missing. Load the matching CUDA toolkit; see README.' >&2
    exit 1
fi
export OPENMM_CUDA_COMPILER="$(command -v nvcc)"
cd "${SLURM_SUBMIT_DIR:?Submit this job from its individual simulation folder}"

if [[ ! -e Data && ! -e Logs ]]; then
    python system_setup.py
else
    python system_setup.py --check
fi
bash run_meld.sh
conda deactivate
