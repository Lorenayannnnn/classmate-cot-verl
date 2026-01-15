set -x

#export TOGETHER_API_KEY="6cf968d54220fa0ee7ff5b256b31e7745bc52e252b71798b731deb2b542d9c56"

# Classmate model: Change url and port number in classmate-cot-verl/outputs/host_classmate_models/classmate_model_mapping.json
#CUDA_VISIBLE_DEVICES=0,1 vllm serve "meta-llama/Llama-3.2-1B-Instruct" --served-model-name "meta-llama/Llama-3.2-1B-Instruct" --tensor_parallel_size 2 --gpu-memory-utilization 0.9 --port 8003
#CUDA_VISIBLE_DEVICES=0 vllm serve "meta-llama/Llama-3.2-1B-Instruct" --served-model-name "meta-llama/Llama-3.2-1B-Instruct" --tensor_parallel_size 1 --gpu-memory-utilization 0.9 --port 8003
# Now using Together AI

data_dir=./data   # run on lambda
#dataset_name="gsm8k_think_prompt"
dataset_name="gsm8k_minimal_answer_box_prompt"
train_path=${data_dir}/${dataset_name}/train.parquet
eval_path=${data_dir}/${dataset_name}/test.parquet
train_files="['$train_path']"
eval_files="['$eval_path']"

#base_model_name_path=Qwen/Qwen3-1.7B
#train_size=7473   # After filtering out too long prompts
base_model_name_path=Qwen/Qwen3-0.6B
train_size=7473   # After filtering out too long prompts

max_response_length=3072

classmate_reward_weight=1
classmate_reward_type=vanilla_truncate_main_classmate_separate
use_classmate_main_cond=always  # always, no_classmate_when_main_incorrect, neg_classmate_when_main_incorrect
#adv_estimator=grpo
adv_estimator=grpo_main_classmate_separated
#adv_estimator=gdpo
#adv_estimator=grpo_main_classmate_separated_non_neg_cl
token_level_classmate_reward_mode=classmate_partial    # classmate_partial, all

gpu_num=8
seed=42
train_batch_size=256
mini_batch_size_per_gpu=32

total_ckpts=28
total_test_times=28

epoch_num=24
rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

#HYDRA_FULL_ERROR=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.qwen_classmate_cot_main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${base_model_name_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((mini_batch_size_per_gpu)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
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
    trainer.experiment_name="${adv_estimator}_${base_model_name_path}_${dataset_name}_${classmate_reward_type}_${use_classmate_main_cond}_classmate_reward_${token_level_classmate_reward_mode}_classmate_llama_${total_episodes}_episodes_seed_${seed}" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=$((train_steps / total_ckpts)) \
    trainer.test_freq=$((train_steps / total_test_times)) \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=${seed} \
    data.return_raw_chat=True \
    reward_model.classmate_cot_reward_configs.classmate_reward_weight=${classmate_reward_weight} \
    reward_model.classmate_cot_reward_configs.classmate_reward_type=${classmate_reward_type} \
    reward_model.classmate_cot_reward_configs.use_classmate_main_cond=${use_classmate_main_cond}
    reward_model.classmate_cot_reward_configs.token_level_classmate_reward_mode=${token_level_classmate_reward_mode}
#    trainer.experiment_name="grpo_${base_model_name_path}_${dataset_name}_with_classmate_reward_llama_${total_episodes}_episodes" \
#    reward_model.sandbox_fusion_url=${sandbox_fusion_url} \
#    reward_model.llm_judge_model=${llm_judge_model}
#    actor_rollout_ref.model.lora_rank=32 \
#    actor_rollout_ref.model.lora_alpha=32 \
#    actor_rollout_ref.model.target_modules=all-linear \
#    actor_rollout_ref.model.use_shm=True
#    actor_rollout_ref.rollout.free_cache_engine=False

#bash my_scripts/qwen3_classmate_gsm8k.sh
