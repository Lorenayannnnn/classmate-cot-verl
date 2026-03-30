#!/bin/sh
#SBATCH --account=hewittlab
#SBATCH --gpus=2
#SBATCH --time=24:00:00
#SBATCH --job-name=olmo_O2

# Mirror key variables from general_reward_classmate_self.sh to compute the output dir
#result_dir="/proj/interaction/interaction-filer/lorena/"
result_dir="outputs/"
dataset_name="general_reward"
seed=2
base_model_name_path=allenai/Olmo-3-7B-Think
train_size=8000
epoch_num=6
rollout_n=8
train_batch_size=16
max_response_length=7168
total_episodes=$((train_size * epoch_num * rollout_n))
main_dir="${result_dir}classmate_cot_w_verl/outputs/${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}_${max_response_length}/OURS_self/seed_${seed}"

mkdir -p "${main_dir}"
exec >"${main_dir}/main_ppo.log" 2>&1

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export HF_HOME=/scratch/hewittlab/lorenayan/.cache/huggingface
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
bash my_scripts/scripts_general_reward/olmo3_8k/slurm_columbiaAI/general_reward_classmate_self.sh

#sbatch my_scripts/scripts_general_reward/olmo3_8k/slurm_columbiaAI/sbatch_files/general_reward_classmate_self.sh
