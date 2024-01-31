#!/bin/bash

#SBATCH --account=cstdl #cstdl laionize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --output=/p/scratch/laionize/franke5/experiments/output/mpi-out.%j
#SBATCH --error=/p/scratch/laionize/franke5/experiments/error/mpi-err.%j
#SBATCH --time=1:00:00  # 6 TODO
#SBATCH --partition=dc-gpu-devel #dc-gpu  #dc-gpu #-devel #booster develbooster dc-gpu "dc-cpu-devel  # 6 TODO
#SBATCH --job-name=fob
###SBATCH --array=0-2

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_LAUNCH_BLOCKING=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export MASTER_PORT=12802
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR


export HF_DATASETS_CACHE="/p/fastdata/mmlaion/franke5/datasets/cache/"
export HF_HOME="/p/fastdata/mmlaion/franke5/model/cache/"
export HF_HUB_OFFLINE="1"

ml Stages/2024
ml CUDA/12
ml GCC/12.3.0
ml Python/3.11.3

source /p/scratch/laionize/franke5/pt21r/bin/activate

export OMP_NUM_THREADS=${SLURM_NTASKS}


# set these variables to select which experiment to run
#first_trial=0
#seed=43

first_trial=9
seed=46

fob_path="/p/scratch/laionize/franke5/workspace/FOB"
workload="imagenet64"
data_dir="/p/fastdata/mmlaion/franke5/model/cache/fob"
output_dir="/p/scratch/laionize/franke5/experiments/fob"
submission="adamcpr"

cd $fob_path

# Running the job
start=`date +%s`
srun python submission_runner.py --workload=$workload --output=$output_dir --data_dir=$data_dir --submission=$submission --hyperparameters "/p/scratch/laionize/franke5/workspace/FOB/icml_related/hyperparameters/adamcpr/imagenet64" --workers=16 --seed $seed --trials 1 --start_trial $((first_trial + SLURM_ARRAY_TASK_ID)) --start_hyperparameter 0 --silent  --devices=4 --max_steps=50000
#srun python submission_runner.py --workload=$workload --output=$output_dir --data_dir=$data_dir --submission=$submission --hyperparameters "/p/scratch/laionize/franke5/workspace/FOB/icml_related/hyperparameters/adamcpr/imagenet64" --workers=16 --seed $seed --trials 1 --start_trial $((first_trial + SLURM_ARRAY_TASK_ID)) --start_hyperparameter $SLURM_ARRAY_TASK_ID --silent  --devices=4 --max_steps=50000
exit_code=$?
end=`date +%s`
runtime=$((end-start))

echo Job execution complete.
echo Runtime: $runtime
exit $exit_code
