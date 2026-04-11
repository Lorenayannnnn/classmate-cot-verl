SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/inference_iterate/iterate_specific_settings_specific_model_seeds.sh
#bash my_eval/scripts/inference_iterate/iterate_specific_settings_specific_model_seeds.sh 180

# ── Shared config ─────────────────────────────────────────────────────────── #
max_tokens=3072
base_model_name_or_path=Qwen/Qwen3-0.6B
think_start_str="<think>"
think_end_str="</think>"
main_cot_keep_rate=1
num_eval_ckpts=6

start_step=${1:-0}     # skip steps below this; pass as $1 to resume mid-run (e.g. 200)

monitor_model_name=gpt-4o-mini
monitor_backend_type=openai        # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker      # "tinker", "vllm_generative", "hf_scoring"
#monitor_model_name=gpt-4o-mini
#monitor_backend_type=openai        # "tinker", "vllm_generative", "hf_scoring"
#llm_judge_model_name=gpt-4.1-mini
#llm_judge_backend_type=openai      # "tinker", "vllm_generative", "hf_scoring"

# ── Per-seed runner ───────────────────────────────────────────────────────── #
# Args: gpu_idx hf_prefix seed dataset_name max_predict_samples
#       step_size max_training_steps run_base
_run_seed_task() {
  local gpu_idx=$1
  local hf_prefix=$2
  local seed=$3
  local dataset_name=$4
  local max_predict_samples=$5
  local step_size=$6
  local max_training_steps=$7
  local run_base=$8

  local last_ckpt=$(( (max_training_steps / step_size) * step_size ))
  local viz_step_size=$(( (last_ckpt / num_eval_ckpts / step_size) * step_size ))
  local viz_offset=$(( last_ckpt - viz_step_size * num_eval_ckpts ))

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
    _run_step "base"
  fi

  for i in $(seq 1 ${num_eval_ckpts}); do
    local step_idx=$((viz_offset + viz_step_size * i))
    [[ ${step_idx} -le ${start_step} ]] && { echo "Skipping step ${step_idx} (<= start_step ${start_step})"; continue; }
    _run_step "${step_idx}"
  done

  local seed_elapsed=$(( SECONDS - seed_start ))
  echo "=== ${main_model_name_or_path} finished in ${seed_elapsed}s ($(( seed_elapsed / 60 ))m $(( seed_elapsed % 60 ))s) ==="
}

# ── Launch tasks (one GPU each) ───────────────────────────────────────────── #
#  GPU  hf_prefix                                                            seed             dataset           samples  step  max   run_base
_run_task 0 LorenaYannnnn/bold_formatting-Qwen3-0.6B-baseline_all_tokens    seed_0         bold_formatting        500      20   420   true  &
_run_task 1 LorenaYannnnn/bold_formatting-Qwen3-0.6B-baseline_all_tokens    seed_1         bold_formatting        500      20   420   true  &
_run_task 2 LorenaYannnnn/bold_formatting-Qwen3-0.6B-baseline_all_tokens    seed_2         bold_formatting        500      20   420   true  &
_run_task 3 LorenaYannnnn/bold_formatting-Qwen3-0.6B-OURS_self              seed_0         bold_formatting        500      20   420   false &
_run_task 4 LorenaYannnnn/bold_formatting-Qwen3-0.6B-OURS_self              seed_1         bold_formatting        500      20   420   false &
_run_task 5 LorenaYannnnn/bold_formatting-Qwen3-0.6B-OURS_self              seed_2         bold_formatting        500      20   420   false &

wait
echo "All tasks finished."