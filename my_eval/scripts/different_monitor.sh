SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -e
export PYTHONPATH=:${PYTHONPATH}

# ── Shared config (mirrors Python arg_parser) ─────────────────────
num_eval_ckpts=6     # number of evenly-distributed checkpoints to evaluate
monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_backend_type=tinker        # "tinker", "vllm_generative", "hf_scoring"
judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker      # "tinker", "vllm_generative", "hf_scoring"
max_new_tokens=3072

# ── Layout ────────────────────────────────────────────────────────
base_model=Qwen3-0.6B
base_root=outputs_eval
seeds=(seed_0 seed_1 seed_2)

# ── Per-task: dataset_name  step_size  methods ────────────────────
# task → dataset_name passed to the verifier/judge
declare -A TASK_DATASET=(
  [confidence]="confidence"
  [sycophancy]="sycophancy"
  [longer_response]="longer_response"
  [general_reward]="general_reward"
  [unsafe_compliance]="unsafe_compliance"
)

# task → total RL training steps (update to match actual runs)
declare -A TASK_MAX_TRAINING_STEPS=(
  [confidence]=210
  [sycophancy]=140
  [longer_response]=270
  [general_reward]=210
  [unsafe_compliance]=210
)

# task → checkpoint save frequency (matches training save_freq cadence)
declare -A TASK_STEP_SIZE=(
  [confidence]=30
  [sycophancy]=20
  [longer_response]=40
  [general_reward]=30
  [unsafe_compliance]=30
)

# task → space-separated list of methods
declare -A TASK_METHODS=(
  [confidence]="baseline_all_tokens OURS_self"
  [sycophancy]="baseline_all_tokens OURS_self"
  [longer_response]="baseline_all_tokens OURS_self"
  [general_reward]="baseline_all_tokens baseline_cot_only OURS_self OURS_llama"
  [unsafe_compliance]="baseline_all_tokens OURS_self"
)

# ── Main loop ─────────────────────────────────────────────────────
for task in confidence sycophancy longer_response general_reward unsafe_compliance; do
  dataset_name=${TASK_DATASET[$task]}
  task_ckpt_step_size=${TASK_STEP_SIZE[$task]}
  task_max_training_steps=${TASK_MAX_TRAINING_STEPS[$task]}
  task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
  task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
  task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))
  read -ra methods <<< "${TASK_METHODS[$task]}"

  do_base_done=false

  for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
      result_dir="${base_root}/${task}/${base_model}/${method}/${seed}"

      # Run --do_base only once per task (base model is shared across all methods/seeds)
      if ! $do_base_done; then
        do_base_arg="--do_base True"
        do_base_done=true
      else
        do_base_arg=""
      fi

      echo "Running ${task} / ${method} / ${seed} ..."
      python ${SRC_DIR}/analysis_module/different_monitor.py \
        --result_dir ${result_dir} \
        --dataset_name ${dataset_name} \
        --max_step_num ${num_eval_ckpts} \
        --step_size ${task_viz_step_size} \
        --viz_offset ${task_viz_offset} \
        --monitor_model_name ${monitor_model_name} \
        --monitor_backend_type ${monitor_backend_type} \
        --judge_model_name ${judge_model_name} \
        --llm_judge_backend_type ${llm_judge_backend_type} \
        --max_new_tokens ${max_new_tokens} \
        ${do_base_arg}
    done
  done
done
