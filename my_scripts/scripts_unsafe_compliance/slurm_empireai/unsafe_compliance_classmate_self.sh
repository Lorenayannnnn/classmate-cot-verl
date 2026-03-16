#!/bin/sh
#SBATCH -A columbia
#SBATCH -p columbia
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --job-name=unsafe_compliance_classmate_self

# Mirror key variables from unsafe_compliance_classmate_self.sh to compute the output dir
result_dir="/proj/interaction/interaction-filer/lorena/"
#result_dir=outputs/
dataset_name="unsafe_compliance"
seed=0
base_model_name_path=Qwen/Qwen3-0.6B
train_size=8000
epoch_num=3
rollout_n=8
train_batch_size=16
total_episodes=$((train_size * epoch_num * rollout_n))
main_dir="${result_dir}classmate_cot_w_verl/outputs/${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}/OURS_self/seed_${seed}"

mkdir -p "${main_dir}"
exec >"${main_dir}/main_ppo.log" 2>&1

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

bash my_scripts/scripts_unsafe_compliance/unsafe_compliance_classmate_self.sh

#sbatch my_scripts/scripts_unsafe_compliance/slurm_empireai/unsafe_compliance_classmate_self.sh