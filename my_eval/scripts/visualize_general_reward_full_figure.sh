SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/visualize_general_reward_full_figure.sh

# ── monitor / judge table (general_reward task only) ───────────────
# Format: "monitor_model  judge_model  use_dynamic_icl  no_explanation"
MONITOR_JUDGE=(
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     false  false"

#  "gpt-4o-mini                           gpt-4.1-mini                         false  false"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507      gpt-4.1-mini                         false  false"
#  "HuggingFaceTB/SmolLM2-360M-Instruct   gpt-4.1-mini                         true   true"
#  "meta-llama/Llama-3.2-1B               gpt-4.1-mini                         true   true"
  "Qwen/Qwen2.5-0.5B-Instruct               gpt-4.1-mini                         true   true"
)

# ── Shared config ──────────────────────────────────────────────────
num_eval_ckpts=6
seeds=seed_0,seed_1,seed_2
base_model=Qwen3-0.6B
base_root=outputs_eval
dataset_split_name=test
dataset=general_reward

task_ckpt_step_size=40
task_max_training_steps=920
task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))

methods_primary="baseline_all_tokens,baseline_all_tokens_w_kl,baseline_cot_only,OURS_self"
methods_secondary="baseline_all_tokens,OURS_self,OURS_llama"
result_dir="${base_root}/${dataset}/${base_model}"

# ── Main loop ──────────────────────────────────────────────────────
for entry in "${MONITOR_JUDGE[@]}"; do
  read -r monitor judge use_dynamic_icl no_explanation <<< "${entry}"
  for behavior_key in sycophancy confidence longer_response general_reward; do
    echo "Visualizing ${dataset} / ${behavior_key} / ${base_model} (primary) [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
      --result_dir         ${result_dir} \
      --dataset_name       ${dataset} \
      --behavior_key       ${behavior_key} \
      --methods            ${methods_primary} \
      --seeds              ${seeds} \
      --max_step_num       ${num_eval_ckpts} \
      --step_size          ${task_viz_step_size} \
      --viz_offset         ${task_viz_offset} \
      --monitor_model_name ${monitor} \
      --judge_model_name   ${judge} \
      --dataset_split_name ${dataset_split_name} \
      $([[ "${use_dynamic_icl}" == "true" ]] && echo "--use_dynamic_icl") \
      $([[ "${no_explanation}" == "true" ]] && echo "--no_explanation")

    echo "Visualizing ${dataset} / ${behavior_key} / ${base_model} (ours comparison) [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/visualize_results_across_models.py \
      --result_dir         ${result_dir} \
      --dataset_name       ${dataset} \
      --behavior_key       ${behavior_key} \
      --methods            ${methods_secondary} \
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
  done
done

echo "Done."
