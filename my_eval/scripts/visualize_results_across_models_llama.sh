SRC_DIR=./my_eval/src
set -e
export PYTHONPATH=:${PYTHONPATH}

# ── Shared config (mirrors Python arg_parser) ─────────────────────
num_eval_ckpts=6     # number of evenly-distributed checkpoints to evaluate
metrics_filename=Qwen_Qwen3-30B-A3B-Instruct-2507_monitor-Qwen_Qwen3-30B-A3B-Instruct-2507_llm_judge_metrics.json
monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
seeds=seed_0,seed_1,seed_2

# ── Layout ────────────────────────────────────────────────────────
base_model=Llama-3.2-3B-Instruct
base_root=outputs_eval

# ── Per-task config (general_reward only for Llama main model) ────
task=general_reward
dataset_name=general_reward
task_ckpt_step_size=40
task_max_training_steps=1500
methods="baseline_all_tokens,baseline_cot_only,OURS_self,OURS_llama"

# ── Main loop ─────────────────────────────────────────────────────
task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))
result_dir="${base_root}/${task}/${base_model}"

# general_reward metrics file contains sub-dicts per behavior key; visualize each separately
for behavior_key in sycophancy confidence longer_response general_reward; do
  echo "Visualizing ${task} / ${behavior_key} / ${base_model} ..."
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
done

#bash my_eval/scripts/visualize_results_across_models_llama.sh
