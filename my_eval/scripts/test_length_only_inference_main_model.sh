SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

#main_model_name_or_path=LorenaYannnnn/20260308-length_only-Qwen3-0.6B_grpo_baseline_192000_episodes_seed_42
main_model_name_or_path=LorenaYannnnn/20260308-length_only-Qwen3-0.6B_OURS_cl_self_partial_192000_episodes_seed_42

base_model_name_or_path=Qwen/Qwen3-0.6B
think_start_str="<think>"
think_end_str="</think>"

main_cot_keep_rate=1

#30, 70, 110, 150, 190, 230, 270

step_size=40

#for i in 0 1 2; do
#gpu_idx=6
for i in 4 5 6; do
gpu_idx=7
#  step_idx=$((step_size*(i+1)))
  step_idx=$((30 + step_size*i))
    if [[ "${i}" == "base" ]]; then
      step_idx="base"
    fi
  (
    echo "Inference main model at step ${step_idx} (${i})"
    CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/inference_main_model.py \
        model_args.model_name_or_path=${main_model_name_or_path} \
        model_args.base_model_name_or_path=${base_model_name_or_path} \
        model_args.main_model_step_idx=${step_idx} \
        model_args.main_cot_keep_rate=${main_cot_keep_rate} \
        running_args.generation_args.max_tokens=3072 \
        data_args.dataset_name=longer_response \
        data_args.max_predict_samples=500 \
        "model_args.think_start_str='${think_start_str}'" \
        "model_args.think_end_str='${think_end_str}'"
    echo "Finished inference main model at step ${step_idx}"
  )
done

#ps -u "$USER" -f | grep -- "./src/main_inference_main_model.py | grep -v grep | awk '{print $2}' | xargs -r kill
#bash my_eval/scripts/test_length_only_inference_main_model.sh

