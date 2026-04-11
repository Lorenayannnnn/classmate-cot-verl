SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

max_tokens=3072

main_model_name_or_path=LorenaYannnnn/20260228-helpfulness-Qwen3-0.6B_grpo_baseline_seed_42_wo_warmup
#main_model_name_or_path=LorenaYannnnn/20260228-helpfulness-Qwen3-0.6B_grpo_OURS_seed_42_wo_warmup

base_model_name_or_path=Qwen/Qwen3-0.6B

main_cot_keep_rate=1

#50, 100, 150, 200, 250, 300, 350

step_size=50

#for i in "base" 0; do
#gpu_idx=0
#for i in 1 2; do
#gpu_idx=1
#for i in 3 4; do
#gpu_idx=2
for i in 5 6; do
gpu_idx=3
  step_idx=$((step_size*(i+1)))
#  step_idx=$((10 + step_size*i))
    if [[ "${i}" == "base" ]]; then
      step_idx="base"
    fi
  (
    echo "Inference main model at step ${step_idx} (${i})"
    CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/main_sycophancy_inference_main_model.py \
        model_args.model_name_or_path=${main_model_name_or_path} \
        model_args.base_model_name_or_path=${base_model_name_or_path} \
        model_args.main_model_step_idx=${step_idx} \
        model_args.main_cot_keep_rate=${main_cot_keep_rate} \
        running_args.generation_args.max_tokens=${max_tokens} \
        data_args.dataset_name=anthropic_hh_rlhf \
        data_args.max_predict_samples=500
    echo "Finished inference main model at step ${step_idx}"
  )
done

#ps -u "$USER" -f | grep -- "./src/main_inference_main_model.py | grep -v grep | awk '{print $2}' | xargs -r kill
#bash my_eval/scripts/compliance_inference_main_model.sh

