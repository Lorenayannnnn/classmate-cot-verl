set -x

export CUDA_HOME=/usr/local/cuda-13.1
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export HF_HOME=/scratch/hewittlab/lorenayan/.cache/huggingface
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

#bash my_scripts/scripts_general_reward/olmo3_8k/slurm_columbiaAI/general_reward_baseline.sh
#srun --account=hewittlab --job-name=olmo_b2 --gres=gpu:2  --pty --time=1:00:00 bash

#result_dir="/proj/interaction/interaction-filer/lorena/"
result_dir=outputs/

data_dir=./data
dataset_name="general_reward"
#seed=0
#gpu_idx=4,5
#seed=1
#gpu_idx=0,1
seed=2
gpu_idx=0,1

train_path=${data_dir}/${dataset_name}/seed_${seed}/train.parquet
eval_path=${data_dir}/${dataset_name}/dev.parquet
train_files="['$train_path']"
eval_files="['$eval_path']"

# </think>
#base_model_name_path=allenai/Olmo-3-7B-Think-SFT
#base_model_name_path=allenai/Olmo-3-7B-Think-DPO
base_model_name_path=allenai/Olmo-3-7B-Think
think_start_str="<think>"
think_end_str=$'</think>\n\n'
token_level_main_reward_mode=all_tokens  # options: "all_tokens", "cot_only", "output_only"

monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_backend_type=tinker
llm_judge_model_name=Skywork/Skywork-Reward-V2-Qwen3-0.6B
llm_judge_backend_type=hf_scoring
llm_judge_backend_dtype=bfloat16
eval_llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
eval_llm_judge_backend_type=tinker

train_size=8000

max_prompt_length=1024
max_response_length=7168
max_num_batched_tokens=$((max_prompt_length + max_response_length))

gpu_num=2
train_batch_size=16
mini_batch_size_per_gpu=16
micro_batch_size_per_gpu=4

save_freq=40
test_freq=20

epoch_num=6
rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

nvidia-smi
nvidia-smi topo -m

[ -z "${SLURM_JOB_ID}" ] && export CUDA_VISIBLE_DEVICES=${gpu_idx}
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${base_model_name_path} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((mini_batch_size_per_gpu)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_num_batched_tokens} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='classmate_cot_w_verl' \
    trainer.experiment_name="${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}_${max_response_length}/baseline_${token_level_main_reward_mode}/seed_${seed}" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=${seed} \
    data.return_raw_chat=True \
    "reward_model.think_start_str='${think_start_str}'" \
    "reward_model.think_end_str='${think_end_str}'" \
    reward_model.monitor_model_name=${monitor_model_name} \
    reward_model.monitor_backend_type=${monitor_backend_type} \
    reward_model.llm_judge_model_name=${llm_judge_model_name} \
    reward_model.llm_judge_backend_type=${llm_judge_backend_type} \
    reward_model.llm_judge_backend_dtype=${llm_judge_backend_dtype} \
    reward_model.eval_llm_judge_model_name=${eval_llm_judge_model_name} \
    reward_model.eval_llm_judge_backend_type=${eval_llm_judge_backend_type} \
    reward_model.token_level_main_reward_mode=${token_level_main_reward_mode} \
    trainer.default_local_dir="${result_dir}"'${trainer.project_name}/outputs/${trainer.experiment_name}' \
    'global_profiler.save_path='"${result_dir}"'${trainer.project_name}/outputs/profile' \
    trainer.validation_data_dir="${result_dir}"'${trainer.project_name}/outputs/${trainer.experiment_name}' \

main_dir="${result_dir}classmate_cot_w_verl/outputs/${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}_${max_response_length}/baseline_${token_level_main_reward_mode}/seed_${seed}"
repo_name="${dataset_name}-${base_model_name_path##*/}_${max_response_length}-baseline_${token_level_main_reward_mode}-seed_${seed}"
python upload_ckpts_to_huggingface.py \
  --root_path ${main_dir} \
  --repo_name ${repo_name}