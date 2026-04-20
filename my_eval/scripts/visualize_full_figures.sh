SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/visualize_full_figures.sh

# ── monitor / judge / dataset table ────────────────────────────────
# dataset: "general_reward" → visualize per behavior key (sycophancy, confidence, longer_response, bold_formatting, general_reward)
#          any other task   → single visualization call
# Format: "monitor_model                  judge_model                               dataset   use_dynamic_icl  no_explanation"
MONITOR_JUDGE_DATASET=(
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507      general_reward       false         false"

  "gpt-4o-mini                          gpt-4.1-mini                          general_reward       false          false"

  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           false         false"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           false         false"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      false         false"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    false         false"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     bold_formatting    false          false"

#  "meta-llama/Llama-3.2-1B              gpt-4.1-mini                          general_reward       true   true"
#  "meta-llama/Llama-3.2-1B                          Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           false          false"
#  "meta-llama/Llama-3.2-1B                          Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           false          false"
#  "meta-llama/Llama-3.2-1B                          Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      false          false"
#  "meta-llama/Llama-3.2-1B                          Qwen/Qwen3-30B-A3B-Instruct-2507     bold_formatting    false          false"

#  "Qwen/Qwen3-30B-A3B-Instruct-2507     gpt-4.1-mini                          general_reward       false  true"
#  "HuggingFaceTB/SmolLM2-360M-Instruct  gpt-4.1-mini                          general_reward       true   true"
#  "Qwen/Qwen2.5-0.5B-Instruct              gpt-4.1-mini                          general_reward       true   true"

#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           true true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           true true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      true true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    true true"
)

# ── Shared config ──────────────────────────────────────────────────
num_eval_ckpts=6
seeds=seed_0,seed_1,seed_2
base_model=Qwen3-0.6B
base_root=outputs_eval
dataset_split_name=test

# ── Per-task config ────────────────────────────────────────────────
declare -A TASK_MAX_TRAINING_STEPS=(
  [general_reward]=920
  [longer_response]=300
  [bold_formatting]=540
  [confidence]=420
  [sycophancy]=200
  [unsafe_compliance]=300
#  [anthropic_sycophancy]=920
)
declare -A TASK_STEP_SIZE=(
  [general_reward]=40
  [longer_response]=20
  [bold_formatting]=20
  [confidence]=20
  [sycophancy]=20
  [unsafe_compliance]=20
#  [anthropic_sycophancy]=40
)
declare -A TASK_METHODS=(
  [general_reward]="baseline_all_tokens,baseline_all_tokens_w_kl,baseline_cot_only,OURS_self"
  [longer_response]="baseline_all_tokens,OURS_self"
  [bold_formatting]="baseline_all_tokens,OURS_self"
  [confidence]="baseline_all_tokens,OURS_self"
  [sycophancy]="baseline_all_tokens,OURS_self"
  [unsafe_compliance]="baseline_all_tokens,OURS_self"
)
declare -A TASK_METHODS_SECONDARY=(
  [general_reward]="baseline_all_tokens,OURS_self,OURS_llama"
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

  if [[ "${dataset}" == "general_reward" ]]; then
    secondary_methods=${TASK_METHODS_SECONDARY[$dataset]}
    for behavior_key in longer_response bold_formatting confidence sycophancy; do
      echo "Visualizing ${dataset} / ${behavior_key} / ${base_model} (primary) [monitor=${monitor}] ..."
      python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
        --result_dir         ${result_dir} \
        --dataset_name       ${dataset} \
        --behavior_key       ${behavior_key} \
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

      if [[ -n "${secondary_methods}" ]]; then
        echo "Visualizing ${dataset} / ${behavior_key} / ${base_model} (ours comparison) [monitor=${monitor}] ..."
        python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
          --result_dir         ${result_dir} \
          --dataset_name       ${dataset} \
          --behavior_key       ${behavior_key} \
          --methods            ${secondary_methods} \
          --seeds              ${seeds} \
          --max_step_num       ${num_eval_ckpts} \
          --step_size          ${task_viz_step_size} \
          --viz_offset         ${task_viz_offset} \
          --monitor_model_name ${monitor} \
          --judge_model_name   ${judge} \
          --figure_suffix      ours_comparison \
          --dataset_split_name ${dataset_split_name} \
      $([[ "${use_dynamic_icl}" == "true" ]] && echo "--use_dynamic_icl") \
      $([[ "${no_explanation}" == "true" ]] && echo "--no_explanation")
      fi
    done
  else
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
  fi
done

echo "Done."
