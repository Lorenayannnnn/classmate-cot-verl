SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/test_different_monitor_general_reward.sh

# ── What to run ────────────────────────────────────────────────────
task=general_reward
method=baseline_all_tokens
seed=seed_0
step_idx=920          # specific RL step to run; set to "base" to run the base checkpoint

# ── Monitor/Judge ──────────────────────────────────────────────────
#monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
#monitor_backend_type=tinker
#judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
#llm_judge_backend_type=tinker
monitor_model_name=gpt-4o-mini
monitor_backend_type=openai
judge_model_name=gpt-4.1-mini
llm_judge_backend_type=openai

max_new_tokens=3072

# ── Layout ─────────────────────────────────────────────────────────
base_model=Qwen3-0.6B
base_root=outputs_eval

# ── Run ────────────────────────────────────────────────────────────
dataset_name=${task}

if [[ "${step_idx}" == "base" ]]; then
  result_dir="${base_root}/${task}/${base_model}/${method}"
  echo "Running ${task} / ${method} / base ..."
  python ${SRC_DIR}/analysis_module/different_monitor.py \
    --result_dir          ${result_dir} \
    --dataset_name        ${dataset_name} \
    --max_step_num        0 \
    --step_size           1 \
    --viz_offset          0 \
    --monitor_model_name  ${monitor_model_name} \
    --monitor_backend_type ${monitor_backend_type} \
    --judge_model_name    ${judge_model_name} \
    --llm_judge_backend_type ${llm_judge_backend_type} \
    --max_new_tokens      ${max_new_tokens} \
    --do_base             True
else
  result_dir="${base_root}/${task}/${base_model}/${method}/${seed}"
  echo "Running ${task} / ${method} / ${seed} / step_${step_idx} ..."
  python ${SRC_DIR}/analysis_module/different_monitor.py \
    --result_dir          ${result_dir} \
    --dataset_name        ${dataset_name} \
    --max_step_num        1 \
    --step_size           1 \
    --viz_offset          $(( step_idx - 1 )) \
    --monitor_model_name  ${monitor_model_name} \
    --monitor_backend_type ${monitor_backend_type} \
    --judge_model_name    ${judge_model_name} \
    --llm_judge_backend_type ${llm_judge_backend_type} \
    --max_new_tokens      ${max_new_tokens}
fi

echo "Done."
