SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/visualize_specific_misbehavior_full_figure.sh

# ── monitor / judge / dataset table ────────────────────────────────
# Format: "monitor_model  judge_model  dataset  use_dynamic_icl  no_explanation"
# dataset: specific-behavior tasks only (confidence, sycophancy, longer_response, unsafe_compliance)
MONITOR_JUDGE_DATASET=(
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           true    true"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           true    true"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      true    true"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    true    true"

  "meta-llama/Llama-3.2-1B              Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           true    true"
  "meta-llama/Llama-3.2-1B              Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           true    true"
  "meta-llama/Llama-3.2-1B              Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      true    true"
  "meta-llama/Llama-3.2-1B              Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    true    true"

#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           true    true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           true    true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      true    true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    true    true"
)

# ── Shared config ──────────────────────────────────────────────────
num_eval_ckpts=6
seeds=seed_0,seed_1,seed_2
base_model=Qwen3-0.6B
base_root=outputs_eval
dataset_split_name=test

# ── Per-task config ────────────────────────────────────────────────
declare -A TASK_MAX_TRAINING_STEPS=(
  [confidence]=420
  [sycophancy]=200
  [longer_response]=300
  [unsafe_compliance]=300
)
declare -A TASK_STEP_SIZE=(
  [confidence]=20
  [sycophancy]=20
  [longer_response]=20
  [unsafe_compliance]=20
)
declare -A TASK_METHODS=(
  [confidence]="baseline_all_tokens,OURS_self"
  [sycophancy]="baseline_all_tokens,OURS_self"
  [longer_response]="baseline_all_tokens,OURS_self"
  [unsafe_compliance]="baseline_all_tokens,OURS_self"
)

# ── Main loop ──────────────────────────────────────────────────────
for entry in "${MONITOR_JUDGE_DATASET[@]}"; do
  read -r monitor judge dataset use_dynamic_icl no_explanation <<< "${entry}"

  task_ckpt_step_size=${TASK_STEP_SIZE[$dataset]}
  task_max_training_steps=${TASK_MAX_TRAINING_STEPS[$dataset]}
  task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
  task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
  task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))
  methods=${TASK_METHODS[$dataset]}
  result_dir="${base_root}/${dataset}/${base_model}"

  echo "Visualizing ${dataset} / ${base_model} [monitor=${monitor}] ..."
  python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
    --result_dir         ${result_dir} \
    --dataset_name       ${dataset} \
    --methods            ${methods} \
    --seeds              ${seeds} \
    --max_step_num       ${num_eval_ckpts} \
    --step_size          ${task_viz_step_size} \
    --viz_offset         ${task_viz_offset} \
    --monitor_model_name ${monitor} \
    --judge_model_name   ${judge} \
    --dataset_split_name ${dataset_split_name} \
    $([[ "${use_dynamic_icl}" == "true" ]] && echo "--use_dynamic_icl") \
    $([[ "${no_explanation}" == "true" ]] && echo "--no_explanation")
done

echo "Done."
