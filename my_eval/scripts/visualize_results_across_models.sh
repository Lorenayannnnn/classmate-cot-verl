SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

# ── Shared config (mirrors Python arg_parser) ─────────────────────
max_step_num=7
step_size=30         # overridden per-task below
metrics_filename=Qwen_Qwen3-30B-A3B-Instruct-2507_monitor-Qwen_Qwen3-30B-A3B-Instruct-2507_llm_judge_metrics.json
monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
seeds=seed_0,seed_1,seed_2

# ── Layout ────────────────────────────────────────────────────────
base_model=Qwen3-0.6B
base_root=outputs_eval/inference_main_model

# ── Per-task config ───────────────────────────────────────────────
declare -A TASK_DATASET=(
  [confidence]="confidence"
  [sycophancy]="sycophancy"
  [longer_response]="longer_response"
  [general_reward]="general_reward"
  [unsafe_compliance]="unsafe_compliance"
)

declare -A TASK_STEP_SIZE=(
  [confidence]=30
  [sycophancy]=20
  [longer_response]=40
  [general_reward]=30
  [unsafe_compliance]=30
)

# Comma-separated methods to compare per task
declare -A TASK_METHODS=(
  [confidence]="baseline_all_tokens,baseline_cot_only,baseline_output_only,OURS_self,OURS_llama"
  [sycophancy]="baseline_all_tokens,baseline_cot_only,baseline_output_only,OURS_self"
  [longer_response]="baseline_all_tokens,baseline_cot_only,baseline_output_only,OURS_self"
  [general_reward]="baseline_all_tokens,baseline_cot_only,baseline_output_only,OURS_self"
  [unsafe_compliance]="baseline_all_tokens,baseline_cot_only,baseline_output_only,OURS_self"
)

# ── Main loop ─────────────────────────────────────────────────────
for task in confidence sycophancy longer_response general_reward unsafe_compliance; do
  dataset_name=${TASK_DATASET[$task]}
  task_step_size=${TASK_STEP_SIZE[$task]}
  methods=${TASK_METHODS[$task]}
  result_dir="${base_root}/${task}/${base_model}"

  echo "Visualizing ${task} / ${base_model} ..."
  python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
    --result_dir ${result_dir} \
    --dataset_name ${dataset_name} \
    --methods ${methods} \
    --seeds ${seeds} \
    --max_step_num ${max_step_num} \
    --step_size ${task_step_size} \
    --metrics_filename ${metrics_filename} \
    --monitor_model_name ${monitor_model_name} \
    --judge_model_name ${judge_model_name}
done

#bash my_eval/scripts/visualize_results_across_models.sh
