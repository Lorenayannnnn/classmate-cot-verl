SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/different_monitor_general_reward.sh

# Set to true to print cost estimate and exit without running anything
estimate_cost=${1:-false}

# ── Shared config (mirrors Python arg_parser) ─────────────────────
num_eval_ckpts=6     # number of evenly-distributed checkpoints to evaluate
#monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
#monitor_backend_type=tinker        # "tinker", "vllm_generative", "hf_scoring"
#judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
#llm_judge_backend_type=tinker      # "tinker", "vllm_generative", "hf_scoring"
monitor_model_name=gpt-4o-mini
monitor_backend_type=openai        # "tinker", "vllm_generative", "hf_scoring"
judge_model_name=gpt-4.1-mini
llm_judge_backend_type=openai      # "tinker", "vllm_generative", "hf_scoring"

#monitor_model_name=gpt-4o
#monitor_backend_type=openai        # "tinker", "vllm_generative", "hf_scoring"
#judge_model_name=gpt-4.1
#llm_judge_backend_type=openai      # "tinker", "vllm_generative", "hf_scoring"
#monitor_model_name=openai/gpt-oss-120b
#monitor_backend_type=tinker        # "tinker", "vllm_generative", "hf_scoring"
#judge_model_name=openai/gpt-oss-120b
#llm_judge_backend_type=tinker      # "tinker", "vllm_generative", "hf_scoring"

max_new_tokens=3072

# ── Layout ────────────────────────────────────────────────────────
base_model=Qwen3-0.6B
base_root=outputs_eval
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

# ── Per-task runner ───────────────────────────────────────────────
# Args: task  method  run_base
_run_task() {
  local task=$1
  local method=$2
  local run_base=$3

  local dataset_name=${TASK_DATASET[$task]}
  local task_ckpt_step_size=${TASK_STEP_SIZE[$task]}
  local task_max_training_steps=${TASK_MAX_TRAINING_STEPS[$task]}
  local task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
  local task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
  local task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))

  # Run base checkpoint once per task (seed-independent; use seeds[0] result_dir)
  if ${run_base}; then
    local base_result_dir="${base_root}/${task}/${base_model}/${method}"
    local src_step_base="${base_result_dir}/step_base"
    echo "Running ${task} / ${method} / base ..."
    python ${SRC_DIR}/analysis_module/different_monitor.py \
      --result_dir ${base_result_dir} \
      --dataset_name ${dataset_name} \
      --max_step_num 0 \
      --step_size 1 \
      --viz_offset 0 \
      --monitor_model_name ${monitor_model_name} \
      --monitor_backend_type ${monitor_backend_type} \
      --judge_model_name ${judge_model_name} \
      --llm_judge_backend_type ${llm_judge_backend_type} \
      --max_new_tokens ${max_new_tokens} \
      --do_base True \
    && {
      # Copy step_base results to all other methods under the same task/base_model/
      for entry in "${TASK_METHOD_RUNBASE[@]}"; do
        local t m rb
        read -r t m rb <<< "${entry}"
        [[ "${t}" != "${task}" ]] && continue   # different task
        [[ "${m}" == "${method}" ]] && continue # same method, skip
        local dst_step_base="${base_root}/${task}/${base_model}/${m}/step_base"
        echo "Copying step_base: ${method} → ${m} ..."
        mkdir -p "${base_root}/${task}/${base_model}/${m}"
        rm -rf "${dst_step_base}"
        cp -r "${src_step_base}" "${dst_step_base}"
      done
    }
  fi

  for seed in "${seeds[@]}"; do
    local result_dir="${base_root}/${task}/${base_model}/${method}/${seed}"
    echo "Running ${task} / ${method} / ${seed} ..."
    python ${SRC_DIR}/analysis_module/different_monitor.py \
      --result_dir ${result_dir} \
      --dataset_name ${dataset_name} \
      --max_step_num ${num_eval_ckpts} \
      --step_size ${task_viz_step_size} \
      --viz_offset ${task_viz_offset} \
      --monitor_model_name ${monitor_model_name} \
      --monitor_backend_type ${monitor_backend_type} \
      --judge_model_name ${judge_model_name} \
      --llm_judge_backend_type ${llm_judge_backend_type} \
      --max_new_tokens ${max_new_tokens} &
  done
  wait
}

# ── task / method / run_base table ───────────────────────────────
TASK_METHOD_RUNBASE=(
  "confidence        baseline_all_tokens  true"
  "confidence        OURS_self            false"
  "sycophancy        baseline_all_tokens  true"
  "sycophancy        OURS_self            false"
  "longer_response   baseline_all_tokens  true"
  "longer_response   OURS_self            false"
  "general_reward    baseline_all_tokens  true"
  "general_reward    baseline_cot_only    false"
  "general_reward    OURS_self            false"
  "general_reward    OURS_llama           false"
  "unsafe_compliance baseline_all_tokens  true"
  "unsafe_compliance OURS_self            false"

#  "confidence        baseline_all_tokens  true"
#  "confidence        OURS_self            true"
#  "sycophancy        baseline_all_tokens  true"
#  "sycophancy        OURS_self            true"
#  "longer_response   baseline_all_tokens  true"
#  "longer_response   OURS_self            true"
#  "general_reward    baseline_all_tokens  true"
#  "general_reward    baseline_cot_only    true"
#  "general_reward    OURS_self            true"
#  "general_reward    OURS_llama           true"
#  "unsafe_compliance baseline_all_tokens  true"
#  "unsafe_compliance OURS_self            true"
)

# ── Cost estimation ───────────────────────────────────────────────
if [[ "${estimate_cost}" == "true" ]]; then
  num_seeds=${#seeds[@]}
  samples_per_dir=500
  avg_tokens_per_sample=2000   # rough estimate: prompt + response tokens combined

  # Approximate blended cost per 1k tokens (input+output) for each model
  declare -A MODEL_COST_PER_1K=(
    [gpt-4o-mini]=0.000375    # $0.15/1M in + $0.60/1M out → avg ~$0.375/1M
    [gpt-4.1-mini]=0.0004
    [gpt-4o]=0.00625          # $2.50/1M in + $10/1M out  → avg ~$6.25/1M
    [gpt-4.1]=0.004           # $2/1M in   + $8/1M out   → avg ~$4/1M
  )
  monitor_cost_per_1k=${MODEL_COST_PER_1K[$monitor_model_name]:-0.005}
  judge_cost_per_1k=${MODEL_COST_PER_1K[$judge_model_name]:-0.005}

  total_rl_dirs=0
  total_base_dirs=0
  echo "══ Per-method breakdown ════════════════════════════════"
  for entry in "${TASK_METHOD_RUNBASE[@]}"; do
    read -r task method run_base <<< "${entry}"
    rl_dirs=$(( num_eval_ckpts * num_seeds ))
    base_dirs=0
    [[ "${run_base}" == "true" ]] && base_dirs=1
    total_rl_dirs=$(( total_rl_dirs + rl_dirs ))
    total_base_dirs=$(( total_base_dirs + base_dirs ))
    printf "  %-25s %-20s  RL dirs: %2d  base: %d\n" "${task}" "${method}" "${rl_dirs}" "${base_dirs}"
  done

  total_dirs=$(( total_rl_dirs + total_base_dirs ))
  total_samples=$(( total_dirs * samples_per_dir ))
  total_tokens_k=$(( total_samples * avg_tokens_per_sample / 1000 ))
  monitor_cost=$(echo "scale=2; ${total_tokens_k} * ${monitor_cost_per_1k}" | bc)
  judge_cost=$(echo "scale=2;   ${total_tokens_k} * ${judge_cost_per_1k}"   | bc)
  total_cost=$(echo "scale=2;   ${monitor_cost} + ${judge_cost}"             | bc)

  echo "══ Summary ═════════════════════════════════════════════"
  echo "  Monitor model      : ${monitor_model_name}"
  echo "  Judge model        : ${judge_model_name}"
  echo "  Total dirs         : ${total_dirs}  (${total_rl_dirs} RL + ${total_base_dirs} base)"
  echo "  Total samples      : ${total_samples}  (${samples_per_dir}/dir)"
  echo "  Avg tokens/sample  : ~${avg_tokens_per_sample}"
  echo "  Monitor cost       : ~\$${monitor_cost}"
  echo "  Judge cost         : ~\$${judge_cost}"
  echo "  Total cost         : ~\$${total_cost}"
  echo "═══════════════════════════════════════════════════════"
  exit 0
fi

# ── Launch all tasks/methods in parallel ──────────────────────────
for entry in "${TASK_METHOD_RUNBASE[@]}"; do
  read -r task method run_base <<< "${entry}"
  _run_task "${task}" "${method}" "${run_base}" &
done

wait
echo "All tasks finished."

#bash my_eval/scripts/different_monitor.sh
#bash my_eval/scripts/different_monitor.sh true   # estimate cost only
