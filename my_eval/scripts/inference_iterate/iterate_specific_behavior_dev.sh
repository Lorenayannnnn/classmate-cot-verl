SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

#bash my_eval/scripts/inference_iterate/iterate_specific_behavior_dev.sh

# ── Shared config ─────────────────────────────────────────────────────────── #
seeds=(seed_0 seed_1 seed_2)
base_model_name_or_path=Qwen/Qwen3-0.6B
think_start_str="<think>"
think_end_str="</think>"
main_cot_keep_rate=1
num_eval_ckpts=6

dataset_split_name=dev

monitor_model_name=gpt-4o-mini
monitor_backend_type=openai              # "tinker", "vllm_generative", "hf_scoring"
llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
llm_judge_backend_type=tinker           # "tinker", "vllm_generative", "hf_scoring"

#monitor_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
#monitor_backend_type=tinker              # "tinker", "vllm_generative", "hf_scoring"
#llm_judge_model_name=Qwen/Qwen3-30B-A3B-Instruct-2507
#llm_judge_backend_type=tinker           # "tinker", "vllm_generative", "hf_scoring"

# ── Per-task runner ───────────────────────────────────────────────────────── #
# Args: gpu_idx hf_prefix dataset_name max_predict_samples
#       step_size max_training_steps run_base
_run_task() {
  local gpu_idx=$1
  local hf_prefix=$2
  local dataset_name=$3
  local max_predict_samples=$4
  local step_size=$5
  local max_training_steps=$6
  local run_base=$7

  local last_ckpt=$(( (max_training_steps / step_size) * step_size ))
  local viz_step_size=$(( (last_ckpt / num_eval_ckpts / step_size) * step_size ))
  local viz_offset=$(( last_ckpt - viz_step_size * num_eval_ckpts ))

  echo "last_ckpt ${last_ckpt}"
  echo "viz_step_size ${viz_step_size}"
  echo "viz_offset ${viz_offset}"

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
        data_args.dataset_split_name=${dataset_split_name} \
        data_args.dataset_name=${dataset_name} \
        data_args.max_predict_samples=${max_predict_samples} \
        "model_args.think_start_str='${think_start_str}'" \
        "model_args.think_end_str='${think_end_str}'"
    echo "Finished inference at step ${step_idx} [${main_model_name_or_path}]"
  }

  # Run base checkpoint once before seeds (base is seed-independent)
  if ${run_base}; then
    local main_model_name_or_path=${hf_prefix}-${seeds[0]}
    _run_step "base"
  fi

  for seed in "${seeds[@]}"; do
    local seed_start=$SECONDS
    local main_model_name_or_path=${hf_prefix}-${seed}
    echo "=== ${main_model_name_or_path} (gpu ${gpu_idx}) ==="

    for i in $(seq 1 ${num_eval_ckpts}); do
      local step_idx=$((viz_offset + viz_step_size * i))
      _run_step "${step_idx}"
    done

    local seed_elapsed=$(( SECONDS - seed_start ))
    echo "=== ${main_model_name_or_path} finished in ${seed_elapsed}s ($(( seed_elapsed / 60 ))m $(( seed_elapsed % 60 ))s) ==="
  done
}

# ── Launch all tasks in parallel (one GPU each) ───────────────────────────── #
#  GPU  model                                                                          dataset           samples  step  max_steps  run_base
#(
#_run_task 0 LorenaYannnnn/confidence-Qwen3-0.6B-baseline_all_tokens       confidence        100  20 420  true
#_run_task 0 LorenaYannnnn/confidence-Qwen3-0.6B-OURS_self                 confidence        100  20 420  false
#) &
#(
#_run_task 1 LorenaYannnnn/longer_response-Qwen3-0.6B-baseline_all_tokens  longer_response   100  20 300  true
#_run_task 1 LorenaYannnnn/longer_response-Qwen3-0.6B-OURS_self            longer_response   100  20 300  false
#) &
#(
#_run_task 2 LorenaYannnnn/sycophancy-Qwen3-0.6B-baseline_all_tokens       sycophancy        100  20 200  true
#_run_task 2 LorenaYannnnn/sycophancy-Qwen3-0.6B-OURS_self                 sycophancy        100  20 200  false
#) &
#(
#_run_task 3 LorenaYannnnn/unsafe_compliance-Qwen3-0.6B-baseline_all_tokens unsafe_compliance 100  20 300  true
#_run_task 3 LorenaYannnnn/unsafe_compliance-Qwen3-0.6B-OURS_self           unsafe_compliance 100  20 300  false
#) &

#_run_task 2 LorenaYannnnn/longer_response-Qwen3-0.6B-baseline_all_tokens  longer_response   100  20 300  true &
#_run_task 3 LorenaYannnnn/longer_response-Qwen3-0.6B-OURS_self            longer_response   100  20 300  false

#wait

# ── Generate dev ICL examples for each task ───────────────────────────────── #
#for task in confidence longer_response sycophancy unsafe_compliance bold_formatting; do
for task in confidence longer_response sycophancy unsafe_compliance; do
  echo "Generating dev ICL examples for ${task} with monitor ${monitor_model_name} and judge ${llm_judge_model_name}..."
  python ${SRC_DIR}/analysis_module/generate_dev_icl_examples.py \
    --base_dir           "outputs_eval/${task}/Qwen3-0.6B" \
    --task               "${task}" \
    --monitor_model_name "${monitor_model_name}" \
    --judge_model_name   "${llm_judge_model_name}" \
    --split              "${dataset_split_name}" \
    --no_explanation
done

echo "All tasks finished."
