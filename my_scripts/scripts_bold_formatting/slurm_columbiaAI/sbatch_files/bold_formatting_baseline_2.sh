#!/bin/sh
#SBATCH --account=hewittlab
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=qwen_b2

# Mirror key variables from bold_formatting_baseline_2.sh to compute the output dir
#result_dir="/proj/interaction/interaction-filer/lorena/"
result_dir="outputs/"
dataset_name="bold_formatting"
seed=2
base_model_name_path=Qwen/Qwen3-0.6B
token_level_main_reward_mode=all_tokens
train_size=8000
epoch_num=3
rollout_n=8
train_batch_size=16
total_episodes=$((train_size * epoch_num * rollout_n))
main_dir="${result_dir}classmate_cot_w_verl/outputs/${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}/baseline_${token_level_main_reward_mode}/seed_${seed}"

mkdir -p "${main_dir}"
exec >"${main_dir}/main_ppo.log" 2>&1

#export PATH=/usr/local/cuda/bin:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#export HF_HOME=/scratch/hewittlab/lorenayan/.cache/huggingface
#unset ROCR_VISIBLE_DEVICES
#unset HIP_VISIBLE_DEVICES
bash my_scripts/scripts_bold_formatting/bold_formatting_baseline_2.sh

#sbatch my_scripts/scripts_bold_formatting/slurm_columbiaAI/sbatch_files/bold_formatting_baseline_2.sh
