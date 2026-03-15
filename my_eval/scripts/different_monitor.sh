SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -e
export PYTHONPATH=:${PYTHONPATH}

# ── Shared config (mirrors Python arg_parser) ─────────────────────
max_step_num=7
step_size=30         # overridden per-task below
monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_source=tinker
judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
judge_source=tinker

# ── Layout ────────────────────────────────────────────────────────
base_model=Qwen3-0.6B
base_root=outputs_eval/inference_main_model
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

# task → step_size (matches training save_freq cadence)
declare -A TASK_STEP_SIZE=(
  [confidence]=30
  [sycophancy]=20
  [longer_response]=40
  [general_reward]=30
  [unsafe_compliance]=30
)

# task → space-separated list of methods
declare -A TASK_METHODS=(
  [confidence]="baseline_all_tokens baseline_cot_only baseline_output_only OURS_self OURS_llama"
  [sycophancy]="baseline_all_tokens baseline_cot_only baseline_output_only OURS_self"
  [longer_response]="baseline_all_tokens baseline_cot_only baseline_output_only OURS_self"
  [general_reward]="baseline_all_tokens baseline_cot_only baseline_output_only OURS_self"
  [unsafe_compliance]="baseline_all_tokens baseline_cot_only baseline_output_only OURS_self"
)

# ── Main loop ─────────────────────────────────────────────────────
for task in confidence sycophancy longer_response general_reward unsafe_compliance; do
  dataset_name=${TASK_DATASET[$task]}
  task_step_size=${TASK_STEP_SIZE[$task]}
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
        --max_step_num ${max_step_num} \
        --step_size ${task_step_size} \
        --monitor_model_name ${monitor_model_name} \
        --monitor_source ${monitor_source} \
        --judge_model_name ${judge_model_name} \
        --judge_source ${judge_source} \
        ${do_base_arg}
    done
  done
done
