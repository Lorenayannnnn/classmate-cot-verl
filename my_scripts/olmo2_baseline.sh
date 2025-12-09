set -x

export TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1"

#TODO change dataset
#data_dir=/data    # run with docker
#data_dir=$HOME/data   # run with conda
data_dir=./data   # run on lambda
dataset_name="RLVR-GSM-MATH-IF-Mixed-Constraints"
train_path=${data_dir}/${dataset_name}/train.parquet
eval_path=${data_dir}/${dataset_name}/eval.parquet
train_size=29674   # After filtering out too long prompts
epoch_num=1    # to make it about 400000 * 8 episodes

train_files="['$train_path']"
eval_files="['$eval_path']"

train_batch_size=256
#train_batch_size=1
mini_batch_size_per_gpu=32
gpu_num=8

total_ckpts=10
total_test_times=10

rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

#base_model_name_path=allenai/OLMo-2-0425-1B-DPO
base_model_name_path=allenai/OLMo-2-1124-7B-DPO

#HYDRA_FULL_ERROR=1
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.main_ppo \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${base_model_name_path} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((mini_batch_size_per_gpu)) \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mini_batch_size_per_gpu} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.1 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
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
    trainer.experiment_name="grpo_${base_model_name_path}_${dataset_name}_baseline_${total_episodes}_episodes" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=$((train_steps / total_ckpts)) \
    trainer.test_freq=$((train_steps / total_test_times)) \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=42 \
#    reward_model.sandbox_fusion_url=${sandbox_fusion_url} \
#    reward_model.llm_judge_model=${llm_judge_model} \
#    actor_rollout_ref.model.lora_rank=32 \
#    actor_rollout_ref.model.lora_alpha=32 \
#    actor_rollout_ref.model.target_modules=all-linear \
#    actor_rollout_ref.model.use_shm=True


#bash my_scripts/olmo2_baseline.sh