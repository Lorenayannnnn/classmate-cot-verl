SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/inference_iterate/llama/iterate_general_reward_cl_llama.sh

# ── Shared config ─────────────────────────────────────────────────────────── #
base_model_name_or_path=meta-llama/Llama-3.2-3B-Instruct
think_start_str="### Reasoning"
think_end_str="### Output"
main_cot_keep_rate=1
num_eval_ckpts=6

step_size=40            # checkpoint save frequency
max_training_steps=1500 # total RL training steps (8000 samples / 16 batch * 3 epochs)

monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
monitor_backend_type=tinker        # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker      # "tinker", "vllm_generative", "hf_scoring"
general_reward_model_name=Skywork/Skywork-Reward-V2-Qwen3-0.6B
general_reward_backend_type=hf_scoring  # "tinker", "vllm_generative", "hf_scoring"

# ── Per-seed runner ───────────────────────────────────────────────────────── #
# Args: gpu_idx hf_prefix seed run_base
_run_seed_task() {
  local gpu_idx=$1
  local hf_prefix=$2
  local seed=$3
  local run_base=$4

  local last_ckpt=$(( (max_training_steps / step_size) * step_size ))
  local viz_step_size=$(( (last_ckpt / num_eval_ckpts / step_size) * step_size ))
  local viz_offset=$(( last_ckpt - viz_step_size * num_eval_ckpts ))

  # Derive output dir components from hf_prefix
  # hf_prefix basename: {task}-{base_model}-{method}  e.g. general_reward-Llama-3.2-3B-Instruct-OURS_llama
  local hf_basename="${hf_prefix##*/}"
  local task_part="${hf_basename%%-*}"                      # general_reward
  local method_str="${hf_basename##*-}"                     # OURS_llama
  local base_model_str="${hf_basename#${task_part}-}"       # Llama-3.2-3B-Instruct-OURS_llama
  base_model_str="${base_model_str%-${method_str}}"         # Llama-3.2-3B-Instruct

  local seed_start=$SECONDS
  local main_model_name_or_path=${hf_prefix}-${seed}
  echo "=== ${main_model_name_or_path} (gpu ${gpu_idx}) ==="

  _run_step() {
    local step_idx=$1
    echo "Inference main model at step ${step_idx} [${main_model_name_or_path}]"
    CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/inference_main_model.py \
        model_args.model_name_or_path=${main_model_name_or_path} \
        model_args.base_model_name_or_path=${base_model_name_or_path} \
        model_args.main_model_step_idx=${step_idx} \
        model_args.main_cot_keep_rate=${main_cot_keep_rate} \
        running_args.generation_args.max_tokens=3072 \
        running_args.monitor_model_name=${monitor_model_name} \
        running_args.monitor_backend_type=${monitor_backend_type} \
        running_args.llm_judge_model_name=${llm_judge_model_name} \
        running_args.llm_judge_backend_type=${llm_judge_backend_type} \
        running_args.general_reward_model_name=${general_reward_model_name} \
        running_args.general_reward_backend_type=${general_reward_backend_type} \
        data_args.dataset_name=general_reward \
        data_args.max_predict_samples=500 \
        "model_args.think_start_str='${think_start_str}'" \
        "model_args.think_end_str='${think_end_str}'"
    echo "Finished inference at step ${step_idx} [${main_model_name_or_path}]"
  }

  if ${run_base}; then
    local log_dir="outputs_eval/${task_part}/${base_model_str}/${method_str}/step_base"
    mkdir -p "${log_dir}"
    _run_step "base" >> "${log_dir}/run.log" 2>&1
  fi

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

# ── Launch all 12 tasks in parallel (one GPU each) ────────────────────────── #
#  GPU  hf_prefix                                                                   seed    run_base
_run_seed_task  0  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-baseline_all_tokens  seed_0  true  &
_run_seed_task  1  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-baseline_all_tokens  seed_1  false &
_run_seed_task  2  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-baseline_all_tokens  seed_2  false &
_run_seed_task  3  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-baseline_cot_only    seed_0  false &
_run_seed_task  4  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-baseline_cot_only    seed_1  false &
_run_seed_task  5  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-baseline_cot_only    seed_2  false &
_run_seed_task  6  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-OURS_self            seed_0  false &
_run_seed_task  7  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-OURS_self            seed_1  false &
_run_seed_task  8  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-OURS_self            seed_2  false &
_run_seed_task  9  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-OURS_llama           seed_0  false &
_run_seed_task 10  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-OURS_llama           seed_1  false &
_run_seed_task 11  LorenaYannnnn/general_reward-Llama-3.2-3B-Instruct-OURS_llama           seed_2  false &

wait
echo "All tasks finished."