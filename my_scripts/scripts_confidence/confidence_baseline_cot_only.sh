set -x

#bash my_scripts/scripts_confidence/confidence_baseline_cot_only.sh

result_dir="/proj/interaction/interaction-filer/lorena/"
#result_dir=outputs

data_dir=./data
dataset_name="confidence"
seed=0
#seed=1
#seed=2
gpu_idx=0

train_path=${data_dir}/${dataset_name}/seed_${seed}/train.parquet
eval_path=${data_dir}/${dataset_name}/dev.parquet
train_files="['$train_path']"
eval_files="['$eval_path']"


#base_model_name_path=Qwen/Qwen3-1.7B
base_model_name_path=Qwen/Qwen3-0.6B
think_start_str="<think>"
think_end_str="</think>"
token_level_main_reward_mode=cot_only  # options: "all", "cot_only", "output_only"
monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_backend_type=tinker  # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker  # "tinker", "vllm_generative", "hf_scoring"
eval_llm_judge_model_name=${llm_judge_model_name}
eval_llm_judge_backend_type=${llm_judge_backend_type}
# TODO change custom_chat_template in verl/trainer/config/model/hf_model.yaml
train_size=8000   # After filtering out too long prompts

max_response_length=3072

gpu_num=1
train_batch_size=16
mini_batch_size_per_gpu=16

#gpu_num=4
#train_batch_size=64
#mini_batch_size_per_gpu=16

#total_ckpts=25
#total_test_times=50
#save_freq=$((train_steps / total_ckpts))
#test_freq=$((train_steps / total_test_times))
save_freq=20
test_freq=10

epoch_num=3
rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

#HYDRA_FULL_ERROR=1
#python3 -m verl.trainer.qwen_main_ppo \
#CUDA_VISIBLE_DEVICES=${gpu_idx} python3 -m verl.trainer.qwen_main_ppo \
[ -z "${SLURM_JOB_ID}" ] && export CUDA_VISIBLE_DEVICES=${gpu_idx}
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${base_model_name_path} \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((mini_batch_size_per_gpu)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='classmate_cot_w_verl' \
    trainer.experiment_name="${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}/baseline_${token_level_main_reward_mode}/seed_${seed}" \
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
    reward_model.eval_llm_judge_model_name=${eval_llm_judge_model_name} \
    reward_model.eval_llm_judge_backend_type=${eval_llm_judge_backend_type} \
    reward_model.token_level_main_reward_mode=${token_level_main_reward_mode} \
    trainer.default_local_dir="${result_dir}"'${trainer.project_name}/outputs/${trainer.experiment_name}' \
    'global_profiler.save_path='"${result_dir}"'${trainer.project_name}/outputs/profile'


main_dir="${result_dir}classmate_cot_w_verl/outputs/${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}/baseline_${token_level_main_reward_mode}/seed_${seed}"
repo_name="${dataset_name}-${base_model_name_path##*/}-baseline_${token_level_main_reward_mode}-seed_${seed}"
python upload_ckpts_to_huggingface.py \
  --root_path ${main_dir} \
  --repo_name ${repo_name}