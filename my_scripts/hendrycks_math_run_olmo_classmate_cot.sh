set -x

#data_dir=/data    # run with docker
data_dir=$HOME/data   # run with conda

#TODO change dataset
#train_path=${data_dir}/gsm8k/train.parquet
#train_size=??    # TODO: get After filtering out too long prompts number
#epoch_num=??    # TODO to make it about 400000 * 8 episodes
#test_path=${data_dir}/gsm8k/test.parquet
#dataset_name="gsm8k"
train_path=${data_dir}/hendrycks_math/train.parquet
train_size=7496   # After filtering out too long prompts
epoch_num=55    # to make it about 400000 * 8 episodes
test_path=${data_dir}/hendrycks_math/test.parquet
dataset_name="hendrycks_math"
#math_train_path=$HOME/data/math/train.parquet
#math_test_path=$HOME/data/math/test.parquet

#train_files="['$gsm8k_train_path', '$math_train_path']"
#test_files="['$gsm8k_test_path', '$math_test_path']"

train_files="['$train_path']"
test_files="['$test_path']"

train_batch_size=512
mini_batch_size_per_gpu=64
gpu_num=4

total_ckpts=40
total_test_times=10

rollout_n=8
train_steps=$((train_size / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

#allenai/OLMo-2-0425-1B-DPO
#allenai/OLMo-2-1124-7B-DPO

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.classmate_cot_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=allenai/OLMo-2-0425-1B-DPO \
    actor_rollout_ref.actor.optim.lr=2e-7 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='classmate_cot_w_verl' \
    trainer.experiment_name="grpo_olmo_1B_${dataset_name}_with_classmate_llama_reward_weight_1_${total_episodes}_episodes" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=$((train_steps / total_ckpts)) \
    trainer.test_freq=$((train_steps / total_test_times)) \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=42


#bash my_scripts/hendrycks_math_run_olmo_classmate_cot.sh