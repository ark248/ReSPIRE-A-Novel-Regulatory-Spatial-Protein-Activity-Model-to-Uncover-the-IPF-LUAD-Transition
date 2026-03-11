#!/bin/bash
#SBATCH --job-name=aracne3
#SBATCH --partition=normal          # adjust if your cluster uses a different GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G

set -euo pipefail

#BLAS settings
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

#activate the conda environment
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"   # change to anaconda3 if that's your install
export CONDA_SOLVER=classic                        # avoids libmamba issues in batch
conda activate myenv
set -u
echo "Activated conda env: myenv"


#run the python script
python -u "/shares/vasciaveo_lab/aarulselvan/arachne/scripts/run_aracne3.py"

echo "Job finished"
