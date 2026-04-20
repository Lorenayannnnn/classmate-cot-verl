SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

export CUDA_HOME=/usr/local/cuda-13.1
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export HF_HOME=/scratch/hewittlab/lorenayan/.cache/huggingface
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

# ── Shared config ─────────────────────────────────────────────────────────── #
max_tokens=3072
base_model_name_or_path=Qwen/Qwen3-0.6B
think_start_str="<think>"
think_end_str="</think>"
main_cot_keep_rate=1
num_eval_ckpts=6
dataset_split_name=test

monitor_model_name=gpt-4o-mini
monitor_backend_type=openai        # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker      # "tinker", "vllm_generative", "hf_scoring"
#llm_judge_model_name=gpt-4.1-mini
#llm_judge_backend_type=openai      # "tinker", "vllm_generative", "hf_scoring"

# ── Per-seed runner ───────────────────────────────────────────────────────── #
# Args: hf_prefix seed dataset_name max_predict_samples step_size max_training_steps run_base
_run_seed_task() {
  local hf_prefix=$1
  local seed=$2
  local dataset_name=$3
  local max_predict_samples=$4
  local step_size=$5
  local max_training_steps=$6
  local run_base=$7

  local last_ckpt=$(( (max_training_steps / step_size) * step_size ))
  local viz_step_size=$(( (last_ckpt / num_eval_ckpts / step_size) * step_size ))
  local viz_offset=$(( last_ckpt - viz_step_size * num_eval_ckpts ))

  # Derive output dir components from hf_prefix
  # hf_prefix basename: {task}-{base_model}-{method}
  local hf_basename="${hf_prefix##*/}"
  local task_part="${hf_basename%%-*}"
  local method_str="${hf_basename##*-}"
  local base_model_str="${hf_basename#${task_part}-}"
  base_model_str="${base_model_str%-${method_str}}"

  local main_model_name_or_path=${hf_prefix}-${seed}
  echo "=== ${main_model_name_or_path} ==="

  _run_step() {
    local step_idx=$1
    echo "Inference main model at step ${step_idx} [${main_model_name_or_path}]"
    python ${SRC_DIR}/inference_main_model.py \
        model_args.model_name_or_path=${main_model_name_or_path} \
        model_args.base_model_name_or_path=${base_model_name_or_path} \
        model_args.main_model_step_idx=${step_idx} \
        model_args.main_cot_keep_rate=${main_cot_keep_rate} \
        running_args.generation_args.max_tokens=${max_tokens} \
        running_args.monitor_model_name=${monitor_model_name} \
        running_args.monitor_backend_type=${monitor_backend_type} \
        running_args.llm_judge_model_name=${llm_judge_model_name} \
        running_args.llm_judge_backend_type=${llm_judge_backend_type} \
        data_args.dataset_split_name=${dataset_split_name} \
        data_args.dataset_name=${dataset_name} \
        data_args.max_predict_samples=${max_predict_samples} \
        "model_args.think_start_str='${think_start_str}'" \
        "model_args.think_end_str='${think_end_str}'"
    echo "Finished inference at step ${step_idx} [${main_model_name_or_path}]"
  }

  if ${run_base}; then
    local log_dir="outputs_eval/${task_part}/${base_model_str}/${method_str}/step_base"
    mkdir -p "${log_dir}"
    _run_step "base" >> "${log_dir}/run.log" 2>&1
  fi

  local seed_start=$SECONDS

  for i in $(seq 1 ${num_eval_ckpts}); do
    local step_idx=$((viz_offset + viz_step_size * i))
    local log_dir="outputs_eval/${task_part}/${base_model_str}/${method_str}/step_${step_idx}/${seed}"
    mkdir -p "${log_dir}"
    _run_step "${step_idx}" >> "${log_dir}/run.log" 2>&1
  done

  local seed_elapsed=$(( SECONDS - seed_start ))
  local seed_dir="outputs_eval/${task_part}/${base_model_str}/${method_str}/${seed}"
  mkdir -p "${seed_dir}"
  echo "${main_model_name_or_path} finished in ${seed_elapsed}s ($(( seed_elapsed / 60 ))m $(( seed_elapsed % 60 ))s)" \
    | tee "${seed_dir}/timing.log"
}

# ── Task table (index → hf_prefix  seed  dataset  samples  step  max_steps  run_base) ── #
# One entry per (model, seed). run_base=true only on the seed_0 row of a model.
# Update --array in the sbatch wrapper to match the number of active (uncommented) entries.
declare -a _TASKS=(
#  "LorenaYannnnn/bold_formatting-Qwen3-0.6B-baseline_all_tokens  seed_0  bold_formatting  500  20  540  true"
#  "LorenaYannnnn/bold_formatting-Qwen3-0.6B-baseline_all_tokens  seed_1  bold_formatting  500  20  540  false"
#  "LorenaYannnnn/bold_formatting-Qwen3-0.6B-baseline_all_tokens  seed_2  bold_formatting  500  20  540  false"
#  "LorenaYannnnn/bold_formatting-Qwen3-0.6B-OURS_self            seed_0  bold_formatting  500  20  540  false"
  "LorenaYannnnn/bold_formatting-Qwen3-0.6B-OURS_self            seed_1  bold_formatting  500  20  540  false"
#  "LorenaYannnnn/bold_formatting-Qwen3-0.6B-OURS_self            seed_2  bold_formatting  500  20  540  false"
)

if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
  # ── SLURM array mode: each job runs exactly one (model, seed) ─────────── #
  read -r _prefix _seed _dataset _samples _step _max_steps _run_base <<< "${_TASKS[${SLURM_ARRAY_TASK_ID}]}"
  _run_seed_task "${_prefix}" "${_seed}" "${_dataset}" "${_samples}" "${_step}" "${_max_steps}" "${_run_base}"
else
  # ── Local mode: run all tasks in parallel ─────────────────────────────── #
  for entry in "${_TASKS[@]}"; do
    read -r _prefix _seed _dataset _samples _step _max_steps _run_base <<< "${entry}"
    _run_seed_task "${_prefix}" "${_seed}" "${_dataset}" "${_samples}" "${_step}" "${_max_steps}" "${_run_base}" &
  done
  wait
  echo "All tasks finished."
fi
