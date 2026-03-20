SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/visualize_results_across_models_specific_misbehavior.sh

# ── Monitor/Judge ─────────────────────
judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_model_name=gpt-4o-mini
metrics_filename=gpt-4o-mini_monitor-Qwen_Qwen3-30B-A3B-Instruct-2507_llm_judge_metrics.json
#monitor_model_name=openai/gpt-oss-120b
#judge_model_name=openai/gpt-oss-120b
#metrics_filename=openai_gpt-oss-120b_monitor-openai_gpt-oss-120b_llm_judge_metrics.json


# ── Shared config (mirrors Python arg_parser) ─────────────────────
num_eval_ckpts=6     # number of evenly-distributed checkpoints to evaluate
seeds=seed_0,seed_1,seed_2

# ── Layout ────────────────────────────────────────────────────────
base_model=Qwen3-0.6B
base_root=outputs_eval

# ── Per-task config ───────────────────────────────────────────────
declare -A TASK_DATASET=(
  [confidence]="confidence"
  [sycophancy]="sycophancy"
  [longer_response]="longer_response"
  [general_reward]="general_reward"
  [unsafe_compliance]="unsafe_compliance"
)

# task → total RL training steps (update to match actual runs)
declare -A TASK_MAX_TRAINING_STEPS=(
  [confidence]=420
  [sycophancy]=200
  [longer_response]=400
  [general_reward]=920
  [unsafe_compliance]=300
)

# task → checkpoint save frequency (matches training save_freq cadence)
declare -A TASK_STEP_SIZE=(
  [confidence]=20
  [sycophancy]=20
  [longer_response]=20
  [general_reward]=40
  [unsafe_compliance]=20
)

# Comma-separated methods to compare per task (main figure)
declare -A TASK_METHODS=(
  [confidence]="baseline_all_tokens,OURS_self"
  [sycophancy]="baseline_all_tokens,OURS_self"
  [longer_response]="baseline_all_tokens,OURS_self"
  [general_reward]="baseline_all_tokens,baseline_cot_only,OURS_self"
  [unsafe_compliance]="baseline_all_tokens,OURS_self"
)

# Secondary methods for an additional figure (general_reward only)
declare -A TASK_METHODS_SECONDARY=(
  [general_reward]="baseline_all_tokens,OURS_self,OURS_llama"
)

# ── Main loop ─────────────────────────────────────────────────────
#for task in confidence sycophancy longer_response general_reward unsafe_compliance; do
for task in confidence sycophancy longer_response unsafe_compliance; do
  dataset_name=${TASK_DATASET[$task]}
  task_ckpt_step_size=${TASK_STEP_SIZE[$task]}
  task_max_training_steps=${TASK_MAX_TRAINING_STEPS[$task]}
  task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
  task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
  task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))
  methods=${TASK_METHODS[$task]}
  result_dir="${base_root}/${task}/${base_model}"

  if [[ "${task}" == "general_reward" ]]; then
    # general_reward metrics file contains sub-dicts per behavior key; visualize each separately
    secondary_methods=${TASK_METHODS_SECONDARY[$task]}
    for behavior_key in sycophancy confidence longer_response general_reward; do
      echo "Visualizing ${task} / ${behavior_key} / ${base_model} (main) ..."
      python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
        --result_dir ${result_dir} \
        --dataset_name ${dataset_name} \
        --behavior_key ${behavior_key} \
        --methods ${methods} \
        --seeds ${seeds} \
        --max_step_num ${num_eval_ckpts} \
        --step_size ${task_viz_step_size} \
        --viz_offset ${task_viz_offset} \
        --metrics_filename ${metrics_filename} \
        --monitor_model_name ${monitor_model_name} \
        --judge_model_name ${judge_model_name}

      if [[ -n "${secondary_methods}" ]]; then
        echo "Visualizing ${task} / ${behavior_key} / ${base_model} (ours comparison) ..."
        python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
          --result_dir ${result_dir} \
          --dataset_name ${dataset_name} \
          --behavior_key ${behavior_key} \
          --methods ${secondary_methods} \
          --seeds ${seeds} \
          --max_step_num ${num_eval_ckpts} \
          --step_size ${task_viz_step_size} \
          --viz_offset ${task_viz_offset} \
          --metrics_filename ${metrics_filename} \
          --monitor_model_name ${monitor_model_name} \
          --judge_model_name ${judge_model_name} \
          --figure_suffix ours_comparison
      fi
    done
  else
    echo "Visualizing ${task} / ${base_model} ..."
    python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
      --result_dir ${result_dir} \
      --dataset_name ${dataset_name} \
      --methods ${methods} \
      --seeds ${seeds} \
      --max_step_num ${num_eval_ckpts} \
      --step_size ${task_viz_step_size} \
      --viz_offset ${task_viz_offset} \
      --metrics_filename ${metrics_filename} \
      --monitor_model_name ${monitor_model_name} \
      --judge_model_name ${judge_model_name}
  fi
done
