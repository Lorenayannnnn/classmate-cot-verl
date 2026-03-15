set -x

# ── [MIND-FACE] ───────────────────────────────────────────────────────────────
# Mind-face baseline training script.
#
# Architecture:
#   - Mind model (frozen): generates CoT from the original prompt via vLLM.
#   - Face model (trainable): receives [prompt + mind_CoT], generates answer.
#   - Reward: scored on face's answer only (mind_face reward manager).
#   - Algorithm: GRPO on face's answer tokens.
#
# Set mind_model_name_or_path to the frozen CoT generator model path.
# Set base_model_name_path to the trainable face model path.
#
# Based on sycophancy_classmate_self.sh.
# ── [END MIND-FACE] ──────────────────────────────────────────────────────────

data_dir=./data   # run on lambda
dataset_name="sycophancy_only"
seed=0
#seed=1
#seed=2
train_path=${data_dir}/${dataset_name}/seed_${seed}/train.parquet
eval_path=${data_dir}/${dataset_name}/dev.parquet
train_files="['$train_path']"
eval_files="['$eval_path']"

# TODO change custom_chat_template in verl/trainer/config/model/hf_model.yaml

# Face model: trainable main policy
base_model_name_path=Qwen/Qwen3-0.6B

# Mind model: frozen CoT generator (passed via classmate_model_name_or_path_list)
# ── [MIND-FACE: CHANGED] mind_model replaces classmate_model ─────────────────
mind_model_name_or_path='["Qwen/Qwen3-0.6B"]'
# ── [END MIND-FACE: CHANGED] ─────────────────────────────────────────────────

monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_backend_type=tinker  # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker  # "tinker", "vllm_generative", "hf_scoring"
eval_llm_judge_model_name=${llm_judge_model_name}
eval_llm_judge_backend_type=${llm_judge_backend_type}

train_size=8000   # After filtering out too long prompts

max_response_length=3072

# ── [MIND-FACE: CHANGED] adv_estimator=grpo (standard GRPO on face tokens) ──
adv_estimator=grpo
# ── [END MIND-FACE: CHANGED] ─────────────────────────────────────────────────

gpu_num=4
train_batch_size=64
mini_batch_size_per_gpu=16

save_freq=10
test_freq=10

epoch_num=3
rollout_n=8
train_steps=$(((train_size + train_batch_size - 1) / train_batch_size * epoch_num))
total_episodes=$((train_size * epoch_num * rollout_n))
gpu_for_train=${gpu_num}

# ── [MIND-FACE: CHANGED] entry point: verl.trainer.mind_face_main_ppo ────────
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.mind_face_main_ppo \
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
    trainer.project_name='mind_face_w_verl' \
    trainer.experiment_name="mind_face_${base_model_name_path}_${dataset_name}_${adv_estimator}_${total_episodes}_episodes_seed_${seed}" \
    trainer.n_gpus_per_node=${gpu_for_train} \
    trainer.nnodes=1 \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${epoch_num} $@ \
    data.seed=${seed} \
    data.return_raw_chat=True \
    reward_model.classmate_cot_reward_configs.classmate_model_name_or_path_list=${mind_model_name_or_path} \
    reward_model.monitor_model_name=${monitor_model_name} \
    reward_model.monitor_backend_type=${monitor_backend_type} \
    reward_model.llm_judge_model_name=${llm_judge_model_name} \
    reward_model.llm_judge_backend_type=${llm_judge_backend_type} \
    reward_model.eval_llm_judge_model_name=${eval_llm_judge_model_name} \
    reward_model.eval_llm_judge_backend_type=${eval_llm_judge_backend_type}
# ── [END MIND-FACE: CHANGED] ─────────────────────────────────────────────────
