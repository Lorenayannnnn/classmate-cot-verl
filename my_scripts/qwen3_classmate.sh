set -x

export TOGETHER_API_KEY="f3fec9e45ab2c98b73b83faaf7a329b07069ed7cbc7655614420b47fda16cab1"

# Classmate model: Change url and port number in classmate-cot-verl/outputs/host_classmate_models/classmate_model_mapping.json
#CUDA_VISIBLE_DEVICES=0,1 vllm serve "meta-llama/Llama-3.2-1B-Instruct" --served-model-name "meta-llama/Llama-3.2-1B-Instruct" --tensor_parallel_size 2 --gpu-memory-utilization 0.9 --port 8003
#CUDA_VISIBLE_DEVICES=0 vllm serve "meta-llama/Llama-3.2-1B-Instruct" --served-model-name "meta-llama/Llama-3.2-1B-Instruct" --tensor_parallel_size 1 --gpu-memory-utilization 0.9 --port 8003
# Now using Together AI

#TODO change dataset
#data_dir=/data    # run with docker
#data_dir=$HOME/data   # run with conda
data_dir=./data   # run on lambda
dataset_name="DeepScaleR"
train_path=${data_dir}/${dataset_name}/train.parquet
eval_path=${data_dir}/${dataset_name}/test.parquet
train_size=40306   # After filtering out too long prompts
epoch_num=1    # to make it about 400000 * 8 episodes

train_files="['$train_path']"
eval_files="['$eval_path']"

train_batch_size=256
mini_batch_size_per_gpu=32
gpu_num=4

total_ckpts=7
total_test_times=7

rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

#base_model_name_path=allenai/OLMo-2-0425-1B-DPO
#base_model_name_path=Qwen/Qwen3-4B-Base
#base_model_name_path=Qwen/Qwen3-4B-Thinking-2507
#base_model_name_path=Qwen/Qwen3-4B
base_model_name_path=Qwen/Qwen3-1.7B

#HYDRA_FULL_ERROR=1
#CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.classmate_cot_main_ppo \
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m verl.trainer.qwen_classmate_cot_main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$eval_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=4906 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${base_model_name_path} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    trainer.experiment_name="grpo_${base_model_name_path}_${dataset_name}_w_classmate_llama_${total_episodes}_episodes" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=$((train_steps / total_ckpts)) \
    trainer.test_freq=$((train_steps / total_test_times)) \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=42
#    trainer.experiment_name="grpo_${base_model_name_path}_${dataset_name}_with_classmate_reward_llama_${total_episodes}_episodes" \
#    reward_model.sandbox_fusion_url=${sandbox_fusion_url} \
#    reward_model.llm_judge_model=${llm_judge_model}
#    actor_rollout_ref.model.lora_rank=32 \
#    actor_rollout_ref.model.lora_alpha=32 \
#    actor_rollout_ref.model.target_modules=all-linear \
#    actor_rollout_ref.model.use_shm=True
#    actor_rollout_ref.rollout.free_cache_engine=False


#bash my_scripts/qwen3_classmate.sh