#!/bin/sh
#SBATCH --account=hewittlab
#SBATCH --exclude=csrtx6000-1,csrtx6000-2
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=bold_inf
#SBATCH --array=0

log_dir="outputs_eval/general_reward/specific_settings_inference_logs"
mkdir -p "${log_dir}"
exec >"${log_dir}/task_${SLURM_ARRAY_TASK_ID}.log" 2>&1

bash my_eval/scripts/inference_iterate/sbatch_files/iterate_specific_settings_specific_model_seeds_helper.sh

#sbatch my_eval/scripts/inference_iterate/sbatch_files/iterate_specific_settings_specific_model_seeds.sh
