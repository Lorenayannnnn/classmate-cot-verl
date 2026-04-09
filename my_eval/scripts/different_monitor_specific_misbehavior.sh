SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/different_monitor_specific_misbehavior.sh
#bash my_eval/scripts/different_monitor_specific_misbehavior.sh true   # estimate cost only

# Set to true to print cost estimate and exit without running anything
estimate_cost=${1:-false}

# ── monitor / judge / dataset table (specific-behavior tasks only) ─
# Format: "monitor_model                             judge_model                dataset      use_icl_demo  overwrite_monitor  overwrite_judge  no_explanation  use_dynamic_icl"
MONITOR_JUDGE_DATASET=(
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           true            true            false           true              true"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           true            true            false           true              true"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      true            true            false           true              true"
  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    true            true            false           true              true"

  "meta-llama/Llama-3.2-1B               Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           true            true            false           true              true"
  "meta-llama/Llama-3.2-1B               Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           true            true            false           true              true"
  "meta-llama/Llama-3.2-1B               Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      true            true            false           true              true"
  "meta-llama/Llama-3.2-1B               Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    true            true            false           true              true"


#  "gpt-4o-mini                          Qwen/Qwen3-30B-A3B-Instruct-2507     bold_formatting    true            true            false           true              true"

#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     confidence           true            true            false           true              true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     sycophancy           true            true            false           true              true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     longer_response      true            true            false           true              true"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507     unsafe_compliance    true            true            false           true              true"
)

# ── task / method / run_base table ────────────────────────────────
TASK_METHOD_RUNBASE=(
  "confidence           baseline_all_tokens  true"
  "confidence           OURS_self            false"
  "sycophancy           baseline_all_tokens  true"
  "sycophancy           OURS_self            false"
  "longer_response      baseline_all_tokens  true"
  "longer_response      OURS_self            false"
  "unsafe_compliance    baseline_all_tokens  true"
  "unsafe_compliance    OURS_self            false"
#  "bold_formatting      baseline_all_tokens  true"
#  "bold_formatting      OURS_self            false"
)

# ── Shared config ──────────────────────────────────────────────────
num_eval_ckpts=6
max_new_tokens=3072
base_model=Qwen3-0.6B
base_root=outputs_eval
dataset_split_name=test
seeds=(seed_0 seed_1 seed_2)

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

# ── Derive backend type from model name ────────────────────────────
_backend_for_model() {
  local model=$1
  if [[ "${model}" == gpt-* ]] || [[ "${model}" == openai/* ]]; then
    echo "openai"
  elif [[ "${model}" == HuggingFaceTB/* ]]; then
    echo "vllm"
  else
    echo "tinker"
  fi
}

# ── Per-task runner ───────────────────────────────────────────────
# Args: task  method  run_base  monitor  monitor_backend  judge  judge_backend  use_icl_demo  overwrite_monitor  overwrite_judge  no_explanation  use_dynamic_icl
_run_task() {
  local task=$1
  local method=$2
  local run_base=$3
  local monitor=$4
  local monitor_backend=$5
  local judge=$6
  local judge_backend=$7
  local use_icl_demo=${8:-false}
  local overwrite_monitor=${9:-false}
  local overwrite_judge=${10:-false}
  local no_explanation=${11:-false}
  local use_dynamic_icl=${12:-false}
  local icl_flag=""
  [[ "${use_icl_demo}" == "true" ]] && icl_flag="--use_ICL_demo"
  local overwrite_monitor_flag=""
  [[ "${overwrite_monitor}" == "true" ]] && overwrite_monitor_flag="--overwrite_monitor"
  local overwrite_judge_flag=""
  [[ "${overwrite_judge}" == "true" ]] && overwrite_judge_flag="--overwrite_judge"
  local no_explanation_flag=""
  [[ "${no_explanation}" == "true" ]] && no_explanation_flag="--no_explanation"
  local use_dynamic_icl_flag=""
  [[ "${use_dynamic_icl}" == "true" ]] && use_dynamic_icl_flag="--use_dynamic_icl"

  local task_ckpt_step_size=${TASK_STEP_SIZE[$task]}
  local task_max_training_steps=${TASK_MAX_TRAINING_STEPS[$task]}
  local task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
  local task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
  local task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))

  if ${run_base}; then
    local base_result_dir="${base_root}/${task}/${base_model}/${method}"
    local src_step_base="${base_result_dir}/step_base"
    echo "Running ${task} / ${method} / base [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/different_monitor.py \
      --result_dir             ${base_result_dir} \
      --dataset_name           ${task} \
      --max_step_num           0 \
      --step_size              1 \
      --viz_offset             0 \
      --monitor_model_name     ${monitor} \
      --monitor_backend_type   ${monitor_backend} \
      --judge_model_name       ${judge} \
      --llm_judge_backend_type ${judge_backend} \
      --max_new_tokens         ${max_new_tokens} \
      --do_base True \
      ${icl_flag} \
      ${no_explanation_flag} \
      ${use_dynamic_icl_flag} \
      ${overwrite_monitor_flag} \
      ${overwrite_judge_flag} \
      --dataset_split_name   ${dataset_split_name} \
    && {
      for entry in "${TASK_METHOD_RUNBASE[@]}"; do
        local t m rb
        read -r t m rb <<< "${entry}"
        [[ "${t}" != "${task}" ]] && continue
        [[ "${m}" == "${method}" ]] && continue
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
    echo "Running ${task} / ${method} / ${seed} [monitor=${monitor}] ..."
    python ${SRC_DIR}/analysis_module/different_monitor.py \
      --result_dir             ${result_dir} \
      --dataset_name           ${task} \
      --max_step_num           ${num_eval_ckpts} \
      --step_size              ${task_viz_step_size} \
      --viz_offset             ${task_viz_offset} \
      --monitor_model_name     ${monitor} \
      --monitor_backend_type   ${monitor_backend} \
      --judge_model_name       ${judge} \
      --llm_judge_backend_type ${judge_backend} \
      --max_new_tokens         ${max_new_tokens} \
      ${icl_flag} \
      ${no_explanation_flag} \
      ${use_dynamic_icl_flag} \
      ${overwrite_monitor_flag} \
      ${overwrite_judge_flag} \
      --dataset_split_name   ${dataset_split_name} &
  done
  wait
}

# ── Cost estimation ────────────────────────────────────────────────
if [[ "${estimate_cost}" == "true" ]]; then
  num_seeds=${#seeds[@]}
  samples_per_dir=500
  avg_tokens_per_sample=2000
  declare -A MODEL_COST_PER_1K=(
    [gpt-4o-mini]=0.000375
    [gpt-4.1-mini]=0.0004
    [gpt-4o]=0.00625
    [gpt-4.1]=0.004
  )

  echo "══ Cost breakdown ══════════════════════════════════════"
  total_cost_all=0
  for mj_entry in "${MONITOR_JUDGE_DATASET[@]}"; do
    read -r monitor judge dataset use_icl_demo overwrite_monitor overwrite_judge no_explanation <<< "${mj_entry}"
    monitor_cost_per_1k=${MODEL_COST_PER_1K[$monitor]:-0}
    judge_cost_per_1k=${MODEL_COST_PER_1K[$judge]:-0}
    total_rl_dirs=0; total_base_dirs=0
    for tmr_entry in "${TASK_METHOD_RUNBASE[@]}"; do
      read -r task method run_base <<< "${tmr_entry}"
      [[ "${task}" != "${dataset}" ]] && continue
      total_rl_dirs=$(( total_rl_dirs + num_eval_ckpts * num_seeds ))
      [[ "${run_base}" == "true" ]] && total_base_dirs=$(( total_base_dirs + 1 ))
    done
    total_dirs=$(( total_rl_dirs + total_base_dirs ))
    total_tokens_k=$(( total_dirs * samples_per_dir * avg_tokens_per_sample / 1000 ))
    pair_cost=$(echo "scale=2; ${total_tokens_k} * (${monitor_cost_per_1k} + ${judge_cost_per_1k})" | bc)
    printf "  monitor=%-35s judge=%-35s dataset=%-20s dirs=%d  cost=~\$%s\n" \
      "${monitor}" "${judge}" "${dataset}" "${total_dirs}" "${pair_cost}"
    total_cost_all=$(echo "scale=2; ${total_cost_all} + ${pair_cost}" | bc)
  done
  echo "══ Total estimated cost: ~\$${total_cost_all} ══════════════"
  exit 0
fi

# ── Launch ────────────────────────────────────────────────────────
for mj_entry in "${MONITOR_JUDGE_DATASET[@]}"; do
  read -r monitor judge dataset use_icl_demo overwrite_monitor overwrite_judge no_explanation use_dynamic_icl <<< "${mj_entry}"
  monitor_backend=$(_backend_for_model "${monitor}")
  judge_backend=$(_backend_for_model "${judge}")
  for tmr_entry in "${TASK_METHOD_RUNBASE[@]}"; do
    read -r task method run_base <<< "${tmr_entry}"
    [[ "${task}" != "${dataset}" ]] && continue
    _run_task "${task}" "${method}" "${run_base}" \
              "${monitor}" "${monitor_backend}" "${judge}" "${judge_backend}" \
              "${use_icl_demo}" "${overwrite_monitor}" "${overwrite_judge}" "${no_explanation}" "${use_dynamic_icl}" &
  done
  wait
done

echo "All tasks finished."
