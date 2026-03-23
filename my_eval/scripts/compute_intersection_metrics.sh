SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/compute_intersection_metrics.sh
#
# Compute intersection metrics across methods (for fair cross-model comparison).
# Run after different_monitor_general_reward.sh has completed.

# ── monitor / judge / dataset table ────────────────────────────────
# Format: "monitor_model  judge_model  dataset"
MONITOR_JUDGE_DATASET=(
  "gpt-4o-mini                          gpt-4.1-mini                          general_reward"
  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507      general_reward"

  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     confidence"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance"

  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     confidence"
  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy"
  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response"
  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance"
)

# ── Shared config ──────────────────────────────────────────────────
num_eval_ckpts=6
seeds=seed_0,seed_1,seed_2
base_model=Qwen3-0.6B
base_root=outputs_eval

# ── Per-task config ────────────────────────────────────────────────
declare -A TASK_MAX_TRAINING_STEPS=(
  [confidence]=420
  [sycophancy]=200
  [longer_response]=400
  [general_reward]=920
  [unsafe_compliance]=300
)
declare -A TASK_STEP_SIZE=(
  [confidence]=20
  [sycophancy]=20
  [longer_response]=20
  [general_reward]=40
  [unsafe_compliance]=20
)
declare -A TASK_METHODS=(
  [confidence]="baseline_all_tokens,OURS_self"
  [sycophancy]="baseline_all_tokens,OURS_self"
  [longer_response]="baseline_all_tokens,OURS_self"
  [general_reward]="baseline_all_tokens,baseline_all_tokens_w_kl,baseline_cot_only,OURS_self"
  [unsafe_compliance]="baseline_all_tokens,OURS_self"
)
declare -A TASK_METHODS_SECONDARY=(
  [general_reward]="baseline_all_tokens,OURS_self,OURS_llama"
)

_calc_steps() {
  local task=$1
  local ckpt_step=${TASK_STEP_SIZE[$task]}
  local max_steps=${TASK_MAX_TRAINING_STEPS[$task]}
  local last_ckpt=$(( (max_steps / ckpt_step) * ckpt_step ))
  task_viz_step_size=$(( (last_ckpt / num_eval_ckpts / ckpt_step) * ckpt_step ))
  task_viz_offset=$(( last_ckpt - task_viz_step_size * num_eval_ckpts ))
}

# ── Main loop ──────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════"
echo " Computing intersection metrics"
echo "════════════════════════════════════════════════════════"

for entry in "${MONITOR_JUDGE_DATASET[@]}"; do
  read -r monitor judge dataset <<< "${entry}"
  _calc_steps ${dataset}
  result_dir="${base_root}/${dataset}/${base_model}"
  methods=${TASK_METHODS[$dataset]}
  secondary_methods=${TASK_METHODS_SECONDARY[$dataset]:-""}

  echo "Computing intersection for ${dataset} / primary (${methods}) [monitor=${monitor}] ..."
  python ${SRC_DIR}/analysis_module/compute_intersection_metrics.py \
    --result_dir         ${result_dir} \
    --dataset_name       ${dataset} \
    --methods            ${methods} \
    --seeds              ${seeds} \
    --max_step_num       ${num_eval_ckpts} \
    --step_size          ${task_viz_step_size} \
    --viz_offset         ${task_viz_offset} \
    --monitor_model_name ${monitor} \
    --judge_model_name   ${judge}

  if [[ -n "${secondary_methods}" ]]; then
    echo "Computing intersection for ${dataset} / secondary (${secondary_methods}) [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/compute_intersection_metrics.py \
      --result_dir         ${result_dir} \
      --dataset_name       ${dataset} \
      --methods            ${secondary_methods} \
      --seeds              ${seeds} \
      --max_step_num       ${num_eval_ckpts} \
      --step_size          ${task_viz_step_size} \
      --viz_offset         ${task_viz_offset} \
      --monitor_model_name ${monitor} \
      --judge_model_name   ${judge}
  fi
done

echo "Done."
