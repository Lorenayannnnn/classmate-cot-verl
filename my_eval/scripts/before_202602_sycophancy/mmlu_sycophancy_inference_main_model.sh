SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

main_model_name_or_path=LorenaYannnnn/20260203-Qwen3-0.6B_mmlu_sycophancy_new_baseline_627984_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260203-Qwen3-0.6B_mmlu_sycophan_new_vanilla_always_cl_partial_deepseek_627984_ep_seed_42
base_model_name_or_path=Qwen/Qwen3-0.6B

main_cot_keep_rate=1

#0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14
#1, 3, 5, 7, 9, 11, 13


step_size=48

#for i in "base" 0; do
#gpu_idx=0
#for i in 2 3; do
#gpu_idx=1
#for i in 4 5; do
#gpu_idx=2
for i in 6 1; do
gpu_idx=3

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
  step_idx=$((step_size*(i+1)))
#  step_idx=$((75 + step_size*i))
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
        data_args.dataset_name=mmlu_sycophancy \
        data_args.max_predict_samples=500
    echo "Finished inference main model at step ${step_idx}"
  )
done

#ps -u "$USER" -f | grep -- "./src/main_inference_main_model.py | grep -v grep | awk '{print $2}' | xargs -r kill
#bash my_eval/scripts/mmlu_sycophancy_inference_main_model.sh

