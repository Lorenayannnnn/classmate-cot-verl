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
max_tokens=7168
base_model_name_or_path=allenai/Olmo-3-7B-Think
think_start_str="<think>"
think_end_str=$'</think>\n\n'
main_cot_keep_rate=1
num_eval_ckpts=6

step_size=40            # checkpoint save frequency (save_freq in training script)
max_training_steps=760 # 8000 samples / 16 batch_size * 6 epochs = 3000 steps

monitor_model_name=gpt-4o-mini
monitor_backend_type=openai        # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=gpt-4.1-mini
llm_judge_backend_type=openai      # "tinker", "vllm_generative", "hf_scoring"
general_reward_model_name=Skywork/Skywork-Reward-V2-Qwen3-0.6B
general_reward_backend_type=hf_scoring  # "tinker", "vllm_generative", "hf_scoring"
dataset_split_name=test

start_step=${1:-0}     # skip steps below this; pass as $1 to resume mid-run (e.g. 200)

# ── Per-task runner ───────────────────────────────────────────────────────── #
# Args: hf_prefix run_base seed [task_start_step]
_run_task() {
  local hf_prefix=$1
  local run_base=$2
  local seed=$3
  local task_start_step=${4:-${start_step}}

  local last_ckpt=$(( (max_training_steps / step_size) * step_size ))
  local viz_step_size=$(( (last_ckpt / num_eval_ckpts / step_size) * step_size ))
  local viz_offset=$(( last_ckpt - viz_step_size * num_eval_ckpts ))

  # Derive output dir components from hf_prefix
  # hf_prefix basename: {task}-{base_model}-{method}  e.g. general_reward-Olmo-3-7B-Think_7168-baseline_all_tokens
  local hf_basename="${hf_prefix##*/}"
  local task_part="${hf_basename%%-*}"                      # general_reward
  local method_str="${hf_basename##*-}"                     # baseline_all_tokens
  local base_model_str="${hf_basename#${task_part}-}"       # Olmo-3-7B-Think_7168-baseline_all_tokens
  base_model_str="${base_model_str%-${method_str}}"         # Olmo-3-7B-Think_7168

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
        running_args.general_reward_model_name=${general_reward_model_name} \
        running_args.general_reward_backend_type=${general_reward_backend_type} \
        data_args.dataset_split_name=${dataset_split_name} \
        data_args.dataset_name=general_reward \
        data_args.max_predict_samples=500 \
        "model_args.think_start_str='${think_start_str}'" \
        "model_args.think_end_str='${think_end_str}'"
    echo "Finished inference at step ${step_idx} [${main_model_name_or_path}]"
  }

  local main_model_name_or_path=${hf_prefix}-${seed}
  echo "=== ${main_model_name_or_path} ==="

  # Run base checkpoint once (seed-independent; only when run_base=true)
  if ${run_base}; then
    local log_dir="outputs_eval/${task_part}/${base_model_str}/${method_str}/step_base"
    mkdir -p "${log_dir}"
    _run_step "base" >> "${log_dir}/run.log" 2>&1
  fi

  local seed_start=$SECONDS

  for i in $(seq 1 ${num_eval_ckpts}); do
    local step_idx=$((viz_offset + viz_step_size * i))
    [[ ${step_idx} -le ${task_start_step} ]] && { echo "Skipping step ${step_idx} (<= start_step ${task_start_step})"; continue; }
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

# ── Task table (index → model  run_base  seed  [start_step]) ─────────────── #
# One entry per (model, seed). run_base=true only on the seed_0 row of a model
# (base checkpoint is seed-independent; runs once using that model path).
# Optional 4th field overrides the global start_step for that task.
declare -a _TASKS=(
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_all_tokens       true   seed_0"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_all_tokens       false  seed_1"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_all_tokens       false  seed_2"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_cot_only         false  seed_0"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_cot_only         false  seed_1"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_cot_only         false  seed_2 390"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-OURS_self                 false  seed_0"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-OURS_self                 false  seed_1 630"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-OURS_self                 false  seed_2  630"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_all_tokens_w_kl  false  seed_0"
  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_all_tokens_w_kl  false  seed_1 630"
#  "LorenaYannnnn/general_reward-Olmo-3-7B-Think_7168-baseline_all_tokens_w_kl  false  seed_2"
)

if [[ -n "${SLURM_ARRAY_TASK_ID}" ]]; then
  # ── SLURM array mode: each job runs exactly one (model, seed) ─────────── #
  read -r _prefix _run_base _seed _task_start_step <<< "${_TASKS[${SLURM_ARRAY_TASK_ID}]}"
  _run_task "${_prefix}" "${_run_base}" "${_seed}" "${_task_start_step}"
else
  # ── Local mode: run all tasks in parallel ─────────────────────────────── #
  for entry in "${_TASKS[@]}"; do
    read -r _prefix _run_base _seed _task_start_step <<< "${entry}"
    _run_task "${_prefix}" "${_run_base}" "${_seed}" "${_task_start_step}" &
  done
  wait
  echo "All tasks finished."
fi