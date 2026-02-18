SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

main_model_name_or_path=LorenaYannnnn/20260216-Qwen3-0.6B_warmup_grpo_baseline_128000_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260216-Qwen3-0.6B_warmup_grpo_OURS_cl_SELF_128000_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260216-Qwen3-0.6B_warmup_grpo_OURS_cl_0.6B_128000_episodes_seed_42

base_model_name_or_path=LorenaYannnnn/20260216-Qwen3-no_nonfactual_irrelevance-0.6B_grpo_warmup_24000_episodes_seed_42

main_cot_keep_rate=1

#10, 20, 30, 40, 50, 60, 70, 80, 90, 100,

step_size=20

#for i in "base" 0 1 2; do
#gpu_idx=0
for i in 3 4 5 6; do
gpu_idx=1

#for i in "base" 0; do
#gpu_idx=0
#for i in 0; do
#gpu_idx=1
#for i in 1; do
#gpu_idx=2
#for i in 2; do
#gpu_idx=3
#for i in 3; do
#gpu_idx=4
#for i in 4; do
#gpu_idx=5
#for i in 5; do
#gpu_idx=6
#for i in 6; do
#  step_idx=$((step_size*(i+1)))
  step_idx=$((10 + step_size*i))
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
        running_args.max_tokens=3072 \
        data_args.dataset_name=helpful_instructions \
        data_args.max_predict_samples=1000
    echo "Finished inference main model at step ${step_idx}"
  )
done

#ps -u "$USER" -f | grep -- "./src/main_inference_main_model.py | grep -v grep | awk '{print $2}' | xargs -r kill
#bash my_eval/scripts/general_sycophancy_inference_main_model.sh

