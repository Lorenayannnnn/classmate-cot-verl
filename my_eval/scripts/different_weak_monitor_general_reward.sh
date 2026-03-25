SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn
set -e
export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/different_weak_monitor_general_reward.sh
#bash my_eval/scripts/different_weak_monitor_general_reward.sh true          # estimate cost only
#bash my_eval/scripts/different_weak_monitor_general_reward.sh false true true    # debug w/ ICL demo: one job, foreground
#bash my_eval/scripts/different_weak_monitor_general_reward.sh false false true  # with ICL demonstrations

# Set to true to print cost estimate and exit without running anything
estimate_cost=${1:-false}
# Set to true to run a single job in the foreground (no & backgrounding) for debugging/breakpoints
debug_mode=${2:-false}
# Set to true to prepend ICL demonstrations to each monitor prompt
use_icl_demo=${3:-false}

# ── Available GPUs ─────────────────────────────────────────────────
#available_gpus=(2 3)
available_gpus=(2)
available_gpu_num=${#available_gpus[@]}

# ── monitor / judge / dataset table (general_reward only) ──────────
MONITOR_JUDGE_DATASET=(
#  "gpt-4o-mini                          gpt-4.1-mini                          general_reward"
#  "Qwen/Qwen3-30B-A3B-Instruct-2507     Qwen/Qwen3-30B-A3B-Instruct-2507      general_reward"

  "HuggingFaceTB/SmolLM2-360M-Instruct     gpt-4.1-mini      general_reward"
#  "meta-llama/Llama-3.2-1B     gpt-4.1-mini      general_reward"
)

# ── task / method / run_base table ────────────────────────────────
TASK_METHOD_RUNBASE=(
  "general_reward    baseline_all_tokens       true"
  "general_reward    baseline_all_tokens_w_kl  false"
  "general_reward    baseline_cot_only         false"
  "general_reward    OURS_self                 false"
  "general_reward    OURS_llama                false"
)

# ── Shared config ──────────────────────────────────────────────────
num_eval_ckpts=6
max_new_tokens=3072
base_model=Qwen3-0.6B
base_root=outputs_eval
seeds=(seed_0 seed_1 seed_2)

# ── Per-task config ────────────────────────────────────────────────
declare -A TASK_MAX_TRAINING_STEPS=(
  [general_reward]=920
)
declare -A TASK_STEP_SIZE=(
  [general_reward]=40
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
    read -r monitor judge dataset <<< "${mj_entry}"
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
    printf "  monitor=%-35s judge=%-35s dirs=%d  cost=~\$%s\n" \
      "${monitor}" "${judge}" "${total_dirs}" "${pair_cost}"
    total_cost_all=$(echo "scale=2; ${total_cost_all} + ${pair_cost}" | bc)
  done
  echo "══ Total estimated cost: ~\$${total_cost_all} ══════════════"
  exit 0
fi

# ── Launch ────────────────────────────────────────────────────────
# Build a flat list of all per-seed jobs: "task method seed monitor monitor_backend judge judge_backend"
# Base steps are handled separately (sequentially) before the per-seed jobs.

for mj_entry in "${MONITOR_JUDGE_DATASET[@]}"; do
  read -r monitor judge dataset <<< "${mj_entry}"
  monitor_backend=$(_backend_for_model "${monitor}")
  judge_backend=$(_backend_for_model "${judge}")

  # ── Step 1: base steps (sequential, one GPU each) ──────────────
  base_method=""
  for tmr_entry in "${TASK_METHOD_RUNBASE[@]}"; do
    read -r task method run_base <<< "${tmr_entry}"
    [[ "${task}" != "${dataset}" ]] && continue
    [[ "${run_base}" != "true" ]] && continue

    task_ckpt_step_size=${TASK_STEP_SIZE[$task]}
    task_max_training_steps=${TASK_MAX_TRAINING_STEPS[$task]}
    task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))

    base_method="${method}"
    base_result_dir="${base_root}/${task}/${base_model}/${method}"
    src_step_base="${base_result_dir}/step_base"
    echo "Running ${task} / ${method} / base [monitor=${monitor}] on GPU ${available_gpus[0]} ..."
    CUDA_VISIBLE_DEVICES=${available_gpus[0]} python ${SRC_DIR}/analysis_module/different_monitor.py \
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
      $([[ "${use_icl_demo}" == "true" ]] && echo "--use_ICL_demo") \
    && {
      for copy_entry in "${TASK_METHOD_RUNBASE[@]}"; do
        read -r ct cm crb <<< "${copy_entry}"
        [[ "${ct}" != "${task}" ]] && continue
        [[ "${cm}" == "${method}" ]] && continue
        dst_step_base="${base_root}/${task}/${base_model}/${cm}/step_base"
        echo "Copying step_base: ${method} → ${cm} ..."
        mkdir -p "${base_root}/${task}/${base_model}/${cm}"
        rm -rf "${dst_step_base}"
        cp -r "${src_step_base}" "${dst_step_base}"
      done
    }
  done

  # ── Step 2: collect all per-seed jobs ──────────────────────────
  per_seed_jobs=()
  for tmr_entry in "${TASK_METHOD_RUNBASE[@]}"; do
    read -r task method run_base <<< "${tmr_entry}"
    [[ "${task}" != "${dataset}" ]] && continue

    task_ckpt_step_size=${TASK_STEP_SIZE[$task]}
    task_max_training_steps=${TASK_MAX_TRAINING_STEPS[$task]}
    task_last_ckpt=$(( (task_max_training_steps / task_ckpt_step_size) * task_ckpt_step_size ))
    task_viz_step_size=$(( (task_last_ckpt / num_eval_ckpts / task_ckpt_step_size) * task_ckpt_step_size ))
    task_viz_offset=$(( task_last_ckpt - task_viz_step_size * num_eval_ckpts ))

    for seed in "${seeds[@]}"; do
      per_seed_jobs+=("${task}|${method}|${seed}|${task_viz_step_size}|${task_viz_offset}")
    done
  done

  # ── Step 3: run per-seed jobs in groups of available_gpu_num ───
  # In debug mode: run only the first job in the foreground (supports breakpoints).
  if [[ "${debug_mode}" == "true" ]]; then
    IFS='|' read -r task method seed task_viz_step_size task_viz_offset <<< "${per_seed_jobs[0]}"
    gpu_id=${available_gpus[0]}
    result_dir="${base_root}/${task}/${base_model}/${method}/${seed}"
    echo "[DEBUG] Running single job: ${task} / ${method} / ${seed} [monitor=${monitor}] on GPU ${gpu_id} ..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python ${SRC_DIR}/analysis_module/different_monitor.py \
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
      $([[ "${use_icl_demo}" == "true" ]] && echo "--use_ICL_demo")
    echo "[DEBUG] Done."
    exit 0
  fi

  total_jobs=${#per_seed_jobs[@]}
  job_idx=0
  while (( job_idx < total_jobs )); do
    group_pids=()
    for (( g=0; g<available_gpu_num && job_idx<total_jobs; g++, job_idx++ )); do
      IFS='|' read -r task method seed task_viz_step_size task_viz_offset <<< "${per_seed_jobs[$job_idx]}"
      gpu_id=${available_gpus[$g]}
      result_dir="${base_root}/${task}/${base_model}/${method}/${seed}"
      echo "Running ${task} / ${method} / ${seed} [monitor=${monitor}] on GPU ${gpu_id} ..."
      CUDA_VISIBLE_DEVICES=${gpu_id} python ${SRC_DIR}/analysis_module/different_monitor.py \
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
        $([[ "${use_icl_demo}" == "true" ]] && echo "--use_ICL_demo") &
      group_pids+=($!)
    done
    # Wait for this group to finish before starting the next
    for pid in "${group_pids[@]}"; do
      wait "${pid}"
    done
  done

done

echo "All tasks finished."