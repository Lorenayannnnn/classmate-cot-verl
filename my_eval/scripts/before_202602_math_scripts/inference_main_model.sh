SRC_DIR=./src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

#outputs_20260105_official_gsm8k_24_epochs
#main_model_name_or_path=LorenaYannnnn/20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_baseline_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260105-Qwen3-0.6B_gsm8k_no_classmate_main_incorrect_1_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260112-Qwen3-0.6B_gsm8k_main_cl_separate_no_cl_main_incorrect_1_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260113-Qwen3-0.6B_gsm8k_dgpo_no_cl_main_wrong_cl_partial_1_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260115-Qwen3-0.6B_gsm8k_m_cl_sep_norm_always_cl_partial_llama_1434816_episodes_seed_42
#base_model_name_or_path=Qwen/Qwen3-0.6B



#To be tested
#LorenaYannnnn/20260117-Qwen3-1.7B-Base_gsm8k_minimal_answer_box_prompt_baseline_1195680_episodes_seed_42
#LorenaYannnnn/20260118-Qwen3-1.7B-Base_gsm8k_m_cl_sep_no_cl_m_wrong_partial_llama_1195680_episodes_seed_42
#LorenaYannnnn/20260117-Qwen3-1.7B-Base_gsm8k_m_cl_separate_always_cl_partial_llama_1195680_episodes_seed_42

#LorenaYannnnn/20260117-Qwen3-1.7B-Base_math_answer_box_baseline_719328_episodes_seed_42
#LorenaYannnnn/20260118-Qwen3-1.7B-Base_MATH_m_cl_sep_no_cl_m_wrong_partial_llama_719328_episodes_seed_42


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
        data_args.dataset_name=openai/gsm8k
#        data_args.max_predict_samples=100
    echo "Finished gsm8k inference main model at step ${step_idx}"

    # hendrycks math
#    dataset_subset_names=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)
#    for subset_name in "${dataset_subset_names[@]}"; do
#    CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/main_inference_main_model.py \
#      model_args.model_name_or_path=${main_model_name_or_path} \
#      model_args.base_model_name_or_path=${base_model_name_or_path} \
#      model_args.main_model_step_idx=${step_idx} \
#      running_args.max_tokens=4096 \
#      data_args.dataset_name=EleutherAI/hendrycks_math \
#      data_args.max_predict_samples=100
#      data_args.dataset_subset_name=${subset_name} \
#    echo "Finished hendrycks math inference main model at step ${step_idx} on subset: ${subset_name}"
#    done
#    echo "Finished hendrycks math inference main model at step ${step_idx}"
#
#    # AIME
#    CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/main_inference_main_model.py \
#        model_args.model_name_or_path=${main_model_name_or_path} \
#        model_args.base_model_name_or_path=${base_model_name_or_path} \
#        model_args.main_model_step_idx=${step_idx} \
#          model_args.max_model_len=${max_model_len} \
#        data_args.dataset_name=AI-MO/aimo-validation-aime \
#        data_args.dataset_split_name=train \
#        data_args.dataset_subset_name=default
#    echo "Finished AIME inference main model at step ${step_idx}"
    echo "Finished inference main model at step ${step_idx}"
  )
done

#ps -u "$USER" -f | grep -- "./src/main_inference_main_model.py | grep -v grep | awk '{print $2}' | xargs -r kill
#bash scripts/inference_main_model.sh












#multiple_of=1
## split 0-9 across 4 gpus: 0-1, 2-3, 4-6, 7-9
#gpu_idx=0
##start_idx=0
##max_idx=1
##start_idx=2
##max_idx=3
##start_idx=4
##max_idx=6
#start_idx=7
#max_idx=9
#for (( i=start_idx; i<=max_idx; i+=multiple_of )); do
#  echo "== Starting group beginning at index ${i} (size up to ${multiple_of}) =="
#  # Launch a parallel job for each j in the current group [i, i+multiple_of-1], clipped to max_idx
#  for (( j=i; j<i+multiple_of && j<=max_idx; j++ )); do
#    # Example step mapping; adjust as needed
#    step_idx=$((31 * (j + 1)))
#    (
#      echo "[j=${j}] Inference main model at step ${step_idx}"
#      CUDA_VISIBLE_DEVICES="${gpu_idx}" python "${SRC_DIR}/main.py" \
#        --base_configs="${CONFIG_DIR}/${base_config}" \
#        --main_model_step_idx "${step_idx}"
#      echo "[j=${j}] Finished inference main model at step ${step_idx}"
#    )
#  done
#  # Wait for all jobs in this group to finish before moving to the next group
#  wait
#  echo "== Finished group starting at ${i} =="
#done



