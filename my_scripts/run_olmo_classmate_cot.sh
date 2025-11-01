set -x


gsm8k_train_path=/data/gsm8k/train.parquet
gsm8k_test_path=/data/gsm8k/test.parquet
#math_train_path=$HOME/data/math/train.parquet
#math_test_path=$HOME/data/math/test.parquet

#train_files="['$gsm8k_train_path', '$math_train_path']"
#test_files="['$gsm8k_test_path', '$math_test_path']"

train_files="['$gsm8k_train_path']"
#7470
test_files="['$gsm8k_test_path']"
#1320

total_episodes=400000
train_size=7470
train_batch_size=512
mini_batch_size_per_gpu=128
gpu_num=4

# Note: Currently train_batch_size = gpu_num * mini_batch_size_per_gpu, which causes ratio between current policy and original ref to always be 1 -> no clipping effect.
total_ckpts=40
total_test_times=10

epoch_num=$((total_episodes / train_size))
gpu_for_train=${gpu_num}
train_steps=$((total_episodes / train_batch_size))

#allenai/OLMo-2-0425-1B-DPO
#allenai/OLMo-2-1124-7B-DPO

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.classmate_cot_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=allenai/OLMo-2-0425-1B-DPO \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((mini_batch_size_per_gpu)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='classmate_cot_w_verl' \
    trainer.experiment_name='grpo_olmo_1B_with_classmate_llama' \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=$((train_steps / total_ckpts)) \
    trainer.test_freq=$((train_steps / total_test_times)) \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=42


#bash my_scripts/run_olmo_classmate_cot.sh