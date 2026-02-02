SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

main_model_name_or_path=LorenaYannnnn/20260117-Qwen3-1.7B-Base_gsm8k_minimal_answer_box_prompt_baseline_1195680_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260118-Qwen3-1.7B-Base_gsm8k_m_cl_sep_no_cl_m_wrong_partial_llama_1195680_episodes_seed_42
base_model_name_or_path=Qwen/Qwen3-1.7B-Base


step_size=84

#for i in "base" 0; do
#gpu_idx=0
#for i in 1 2; do
#gpu_idx=1
#for i in 3 4; do
#gpu_idx=2
for i in 5 6; do
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
    # gsm8k
    CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/main_inference_main_model.py \
        model_args.model_name_or_path=${main_model_name_or_path} \
        model_args.base_model_name_or_path=${base_model_name_or_path} \
        model_args.main_model_step_idx=${step_idx} \
        running_args.max_tokens=3072 \
        data_args.dataset_name=mmlu_sycophancy
        data_args.max_predict_samples=100
    echo "Finished inference main model at step ${step_idx}"
  )
done

#ps -u "$USER" -f | grep -- "./src/main_inference_main_model.py | grep -v grep | awk '{print $2}' | xargs -r kill
#bash my_eval/scripts/mmlu_sycophancy_inference_main_model.sh

