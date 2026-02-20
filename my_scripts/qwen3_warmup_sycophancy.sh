set -x

#export HF_HOME=/scratch/hewittlab/lorenayan/.cache/huggingface
#unset ROCR_VISIBLE_DEVICES
#unset HIP_VISIBLE_DEVICES

#export TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1"

data_dir=./data   # run on lambda
dataset_name="helpful_instructions"
#dataset_name="anthropic_hh_rlhf"

train_path=${data_dir}/${dataset_name}/warmup_train.parquet
eval_path=${data_dir}/${dataset_name}/warmup_dev.parquet
train_files="['$train_path']"
eval_files="['$eval_path']"


#base_model_name_path=Qwen/Qwen3-1.7B
base_model_name_path=Qwen/Qwen3-0.6B
# TODO change custom_chat_template in verl/trainer/config/model/hf_model.yaml
#base_model_name_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
#base_model_name_path=Qwen/Qwen2.5-Math-1.5B
#base_model_name_path=Qwen/Qwen3-0.6B-Base
train_size=1000   # After filtering out too long prompts

max_response_length=3072

gpu_num=4
seed=42
train_batch_size=64
mini_batch_size_per_gpu=16

total_ckpts=10
total_test_times=10
#save_freq=$((train_steps / total_ckpts))
#test_freq=$((train_steps / total_test_times))
save_freq=5
test_freq=5

epoch_num=2
rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

#HYDRA_FULL_ERROR=1
#python3 -m verl.trainer.qwen_main_ppo \
#CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.qwen_main_ppo \
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.qwen_main_ppo \
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
    trainer.experiment_name="${dataset_name}_${base_model_name_path}_grpo_warmup_${total_episodes}_episodes_seed_${seed}" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=${seed} \
    data.return_raw_chat=True
#    reward_model.sandbox_fusion_url=${sandbox_fusion_url} \
#    reward_model.llm_judge_model=${llm_judge_model} \
#    actor_rollout_ref.model.lora_rank=32 \
#    actor_rollout_ref.model.lora_alpha=32 \
#    actor_rollout_ref.model.target_modules=all-linear \
#    actor_rollout_ref.model.use_shm=True

#bash my_scripts/qwen3_warmup_sycophancy.sh