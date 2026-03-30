SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/visualize_main_figure.sh

# ── monitor / judge / dataset table ────────────────────────────────
# Format: "monitor_model  judge_model  dataset"
# dataset: "general_reward"      → Group 1: 3 behaviors + reward score, two method comparisons
#          "specific_behaviors"  → Group 2: one figure with all 4 behaviors, each column from
#                                   the model trained on that specific behavior
MONITOR_JUDGE_DATASET=(
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507      general_reward"

#  "gpt-4o-mini                          gpt-4.1-mini                          general_reward"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     gpt-4.1-mini                          general_reward"
#  "HuggingFaceTB/SmolLM2-360M-Instruct  gpt-4.1-mini                          general_reward"
#  "meta-llama/Llama-3.2-1B              gpt-4.1-mini                          general_reward"

  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507      specific_behaviors"
  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507      specific_behaviors"
)

# ── Shared config ──────────────────────────────────────────────────
num_eval_ckpts=6
seeds=seed_0,seed_1,seed_2
base_model=Qwen3-0.6B
base_root=outputs_eval
dataset_split_name=test

# ── Per-task step config ───────────────────────────────────────────
declare -A TASK_MAX_TRAINING_STEPS=(
  [confidence]=420
  [sycophancy]=200
  [longer_response]=300
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

_calc_steps() {
  local task=$1
  local ckpt_step=${TASK_STEP_SIZE[$task]}
  local max_steps=${TASK_MAX_TRAINING_STEPS[$task]}
  local last_ckpt=$(( (max_steps / ckpt_step) * ckpt_step ))
  task_viz_step_size=$(( (last_ckpt / num_eval_ckpts / ckpt_step) * ckpt_step ))
  task_viz_offset=$(( last_ckpt - task_viz_step_size * num_eval_ckpts ))
}

# ── Main loop ──────────────────────────────────────────────────────
for entry in "${MONITOR_JUDGE_DATASET[@]}"; do
  read -r monitor judge dataset <<< "${entry}"

  if [[ "${dataset}" == "general_reward" ]]; then
    # Group 1: 2×4 figure — row 0 = RMSE (3 behaviors + reward score)
    #                        row 1 = ground truth mean (3 behaviors + empty)
    _calc_steps general_reward
    result_dir="${base_root}/general_reward/${base_model}"

    echo "Generating main figure: general_reward (primary) [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/visualize_main_figure.py \
      --result_dir         ${result_dir} \
      --dataset_name       general_reward \
      --behavior_keys      longer_response,confidence,sycophancy \
      --include_reward_score \
      --methods            baseline_all_tokens,baseline_all_tokens_w_kl,baseline_cot_only,OURS_self \
      --seeds              ${seeds} \
      --max_step_num       ${num_eval_ckpts} \
      --step_size          ${task_viz_step_size} \
      --viz_offset         ${task_viz_offset} \
      --monitor_model_name ${monitor} \
      --judge_model_name   ${judge} \
      --dataset_split_name ${dataset_split_name}

    echo "Generating main figure: general_reward (ours comparison) [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/visualize_main_figure.py \
      --result_dir         ${result_dir} \
      --dataset_name       general_reward \
      --behavior_keys      longer_response,confidence,sycophancy \
      --include_reward_score \
      --methods            baseline_all_tokens,OURS_self,OURS_llama \
      --seeds              ${seeds} \
      --max_step_num       ${num_eval_ckpts} \
      --step_size          ${task_viz_step_size} \
      --viz_offset         ${task_viz_offset} \
      --monitor_model_name ${monitor} \
      --judge_model_name   ${judge} \
      --figure_suffix      ours_comparison \
      --dataset_split_name ${dataset_split_name}

  elif [[ "${dataset}" == "specific_behaviors" ]]; then
    # Group 2: 2×4 figure — each column is a different behavior-specific trained model
    # Build per-behavior result_dirs and step configs
    per_bk_dirs=()
    per_bk_steps=()
    for bk in longer_response confidence sycophancy unsafe_compliance; do
      _calc_steps ${bk}
      per_bk_dirs+=("${bk}:${base_root}/${bk}/${base_model}")
      per_bk_steps+=("${bk}:${num_eval_ckpts}:${task_viz_step_size}:${task_viz_offset}")
    done

    echo "Generating main figure: specific_behaviors [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/visualize_main_figure.py \
      --dataset_name       specific_behaviors \
      --behavior_keys      longer_response,confidence,sycophancy,unsafe_compliance \
      --methods            baseline_all_tokens,OURS_self \
      --seeds              ${seeds} \
      --monitor_model_name ${monitor} \
      --judge_model_name   ${judge} \
      --per_behavior_result_dirs  "${per_bk_dirs[@]}" \
      --per_behavior_step_configs "${per_bk_steps[@]}" \
      --dataset_split_name ${dataset_split_name}
  fi
done

echo "Done."
