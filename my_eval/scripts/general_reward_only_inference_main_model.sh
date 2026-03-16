SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_all_tokens-seed_0
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_all_tokens-seed_1
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_all_tokens-seed_2
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_self-seed_0
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_self-seed_1
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_self-seed_2

base_model_name_or_path=Qwen/Qwen3-0.6B
think_start_str="<think>"
think_end_str="</think>"

main_cot_keep_rate=1

step_size=30

#for i in "base" 0; do
#gpu_idx=4
#for i in 1 2; do
#gpu_idx=5
#for i in 3 4; do
#gpu_idx=6
for i in 5 6; do
gpu_idx=7
  step_idx=$((step_size*(i+1)))
#  step_idx=$((10 + step_size*i))
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
        data_args.dataset_name=confidence \
        data_args.max_predict_samples=500 \
        "reward_model.think_start_str='${think_start_str}'" \
        "reward_model.think_end_str='${think_end_str}'"
    echo "Finished inference main model at step ${step_idx}"
  )
done

#ps -u "$USER" -f | grep -- "./src/main_inference_main_model.py | grep -v grep | awk '{print $2}' | xargs -r kill
#bash my_eval/scripts/confidence_only_inference_main_model.sh

