#!/bin/sh
#SBATCH --account=hewittlab
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --job-name=gr_dev_inf
#SBATCH --array=0-4

log_dir="outputs_eval/general_reward/dev_inference_logs"
mkdir -p "${log_dir}"
exec >"${log_dir}/task_${SLURM_ARRAY_TASK_ID}.log" 2>&1

export CUDA_HOME=/usr/local/cuda-13.1
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export HF_HOME=/scratch/hewittlab/lorenayan/.cache/huggingface
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
bash my_eval/scripts/inference_iterate/sbatch_files/iterate_general_reward_dev_helper.sh

#sbatch my_eval/scripts/inference_iterate/sbatch_files/iterate_general_reward_dev.sh
