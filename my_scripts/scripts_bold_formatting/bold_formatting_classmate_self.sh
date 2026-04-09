set -x

#bash my_scripts/scripts_bold_formatting/bold_formatting_classmate_self.sh

#result_dir="/proj/interaction/interaction-filer/lorena/"
result_dir=outputs/

data_dir=./data
dataset_name="bold_formatting"
seed=0
gpu_idx=0

train_path=${data_dir}/${dataset_name}/seed_${seed}/train.parquet
eval_path=${data_dir}/${dataset_name}/dev.parquet
train_files="['$train_path']"
eval_files="['$eval_path']"

# TODO change custom_chat_template in verl/trainer/config/model/hf_model.yaml
#TODO Change classmate_model_name_or_path_list in qwen_classmate_cot_ppo_trainer.yaml
base_model_name_path=Qwen/Qwen3-0.6B
classmate_model_name_or_path_list='["Qwen/Qwen3-0.6B"]'
think_start_str="<think>"
think_end_str="</think>"
monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_backend_type=tinker  # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker  # "tinker", "vllm_generative", "hf_scoring"
eval_llm_judge_model_name=${llm_judge_model_name}
eval_llm_judge_backend_type=${llm_judge_backend_type}

train_size=8000   # After filtering out too long prompts

max_response_length=3072

classmate_reward_weight=1
classmate_reward_type=vanilla_reward
#use_classmate_main_cond=always  # no_classmate, always, no_classmate_when_main_incorrect, neg_classmate_when_main_incorrect
adv_estimator=grpo_w_classmate
#adv_estimator=gdpo
#adv_estimator=gdpo_wo_bn
token_level_classmate_reward_mode=classmate_partial    # classmate_partial, all
#main_cot_keep_rate=0.7
# add_consistency_reward=False

gpu_num=1
train_batch_size=16
mini_batch_size_per_gpu=16

#total_ckpts=25
#total_test_times=50
#save_freq=$((train_steps / total_ckpts))
#test_freq=$((train_steps / total_test_times))
#save_freq=10
#test_freq=10
save_freq=20
test_freq=10

epoch_num=3
rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

#HYDRA_FULL_ERROR=1
#[ -z "${SLURM_JOB_ID}" ] && export CUDA_VISIBLE_DEVICES=${gpu_idx}
python3 -m verl.trainer.classmate_cot_main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
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
    trainer.experiment_name="${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}/OURS_self/seed_${seed}" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=${seed} \
    data.return_raw_chat=True \
    reward_model.classmate_cot_reward_configs.classmate_reward_weight=${classmate_reward_weight} \
    reward_model.classmate_cot_reward_configs.classmate_reward_type=${classmate_reward_type} \
    reward_model.classmate_cot_reward_configs.token_level_classmate_reward_mode=${token_level_classmate_reward_mode} \
    reward_model.classmate_cot_reward_configs.classmate_model_name_or_path_list=${classmate_model_name_or_path_list} \
    "reward_model.think_start_str='${think_start_str}'" \
    "reward_model.think_end_str='${think_end_str}'" \
    reward_model.monitor_model_name=${monitor_model_name} \
    reward_model.monitor_backend_type=${monitor_backend_type} \
    reward_model.llm_judge_model_name=${llm_judge_model_name} \
    reward_model.llm_judge_backend_type=${llm_judge_backend_type} \
    reward_model.eval_llm_judge_model_name=${eval_llm_judge_model_name} \
    reward_model.eval_llm_judge_backend_type=${eval_llm_judge_backend_type} \
    trainer.default_local_dir="${result_dir}"'${trainer.project_name}/outputs/${trainer.experiment_name}' \
    'global_profiler.save_path='"${result_dir}"'${trainer.project_name}/outputs/profile'

#    reward_model.classmate_cot_reward_configs.add_consistency_reward=${add_consistency_reward}

#    reward_model.classmate_cot_reward_configs.use_classmate_main_cond=${use_classmate_main_cond} \
#    trainer.experiment_name="${adv_estimator}_${base_model_name_path}_${dataset_name}_${classmate_reward_type}_keep_m_${main_cot_keep_rate}_${use_classmate_main_cond}_cl_${token_level_classmate_reward_mode}_classmate_llama_${total_episodes}_episodes_seed_${seed}" \
#    reward_model.classmate_cot_reward_configs.main_cot_keep_rate=${main_cot_keep_rate} \
#    trainer.save_freq=$((train_steps / total_ckpts)) \
#    trainer.test_freq=$((train_steps / total_test_times)) \
#    trainer.experiment_name="grpo_${base_model_name_path}_${dataset_name}_with_classmate_reward_llama_${total_episodes}_episodes" \
#    reward_model.sandbox_fusion_url=${sandbox_fusion_url} \
#    reward_model.llm_judge_model=${llm_judge_model}
#    actor_rollout_ref.model.lora_rank=32 \
#    actor_rollout_ref.model.lora_alpha=32 \
#    actor_rollout_ref.model.target_modules=all-linear \
#    actor_rollout_ref.model.use_shm=True
#    actor_rollout_ref.rollout.free_cache_engine=False


main_dir="${result_dir}classmate_cot_w_verl/outputs/${dataset_name}/grpo_${total_episodes}_episodes/${base_model_name_path}/OURS_self/seed_${seed}"
repo_name="${dataset_name}-${base_model_name_path##*/}-OURS_self-seed_${seed}"
python upload_ckpts_to_huggingface.py \
  --root_path ${main_dir} \
  --repo_name ${repo_name}