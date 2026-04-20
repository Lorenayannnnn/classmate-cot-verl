#!/bin/sh
#SBATCH --account=hewittlab
#SBATCH --exclude=csrtx6000-1,csrtx6000-2
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --job-name=olmoinf

log_dir="outputs_eval/general_reward/olmo3_inference_logs"
mkdir -p "${log_dir}"
exec >"${log_dir}/task_${SLURM_ARRAY_TASK_ID}.log" 2>&1

bash my_eval/scripts/inference_iterate/sbatch_files/iterate_general_reward_olmo3_helper_1.sh

#sbatch my_eval/scripts/inference_iterate/sbatch_files/iterate_general_reward_olmo3_1.sh
