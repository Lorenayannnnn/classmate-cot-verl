SRC_DIR=./src

export VLLM_WORKER_MULTIPROC_METHOD=spawn
#export TOGETHER_API_KEY="6cf968d54220fa0ee7ff5b256b31e7745bc52e252b71798b731deb2b542d9c56"

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

### START EDITING HERE ###

#main_model_name_or_path=LorenaYannnnn/20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_baseline_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260105-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260105-Qwen3-0.6B_gsm8k_no_classmate_main_incorrect_1_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260112-Qwen3-0.6B_gsm8k_main_cl_separate_no_cl_main_incorrect_1_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260113-Qwen3-0.6B_gsm8k_dgpo_no_cl_main_wrong_cl_partial_1_llama_1434816_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260115-Qwen3-0.6B_gsm8k_m_cl_sep_norm_always_cl_partial_llama_1434816_episodes_seed_42


main_model_name_or_path=LorenaYannnnn/20260117-Qwen3-1.7B-Base_gsm8k_minimal_answer_box_prompt_baseline_1195680_episodes_seed_42
#main_model_name_or_path=LorenaYannnnn/20260118-Qwen3-1.7B-Base_gsm8k_m_cl_sep_no_cl_m_wrong_partial_llama_1195680_episodes_seed_42
base_model_name_or_path=Qwen/Qwen3-1.7B-Base

step_size=84

for i in "base" 0; do
gpu_idx=0
#for i in 1 2; do
#gpu_idx=1
#for i in 3 4; do
#gpu_idx=2
#for i in 5 6; do
#gpu_idx=3

##for i in "base" 0; do
##gpu_idx=0
##for i in 0; do
##gpu_idx=1
##for i in 1; do
##gpu_idx=2
##for i in 2; do
##gpu_idx=3
##for i in 3; do
##gpu_idx=4
##for i in 4; do
##gpu_idx=5
##for i in 5; do
#gpu_idx=6
#for i in 6; do
  (
#    step_idx=$((75 + step_size*i))
    step_idx=$((step_size*(i+1)))
    if [[ "${i}" == "wo_cot" ]]; then
      wo_cot=True
      step_idx="wo_cot"
    elif [[ "${i}" == "base" ]]; then   #allenai/OLMo-2-0425-1B-DPO
      step_idx="base"
      wo_cot=False
    else
      wo_cot=False
    fi
#    for model_name_or_path in "meta-llama/Llama-3.2-1B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "allenai/OLMo-2-0425-1B-Instruct"; do
    for model_name_or_path in "meta-llama/Llama-3.2-1B-Instruct"; do
      echo "Start running classmate ${model_name_or_path} on ${main_model_name_or_path} step ${step_idx}"
#       gsm8k
      CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/main_classmate_utility.py \
          model_args.model_name_or_path=${model_name_or_path} \
          data_args.main_model_name_or_path=${main_model_name_or_path} \
          data_args.main_model_step_idx=${step_idx} \
          data_args.wo_cot=${wo_cot} \
          data_args.dataset_name=openai/gsm8k
#          data_args.max_predict_samples=200
      echo "Finished running classmate ${model_name_or_path} on gsm8k on ${main_model_name_or_path} step ${step_idx}"

      # hendrycks math
#      dataset_subset_names=(algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus)
#      for subset_name in "${dataset_subset_names[@]}"; do
#      CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/main_classmate_utility.py \
#        model_args.model_name_or_path=${model_name_or_path} \
#        data_args.main_model_name_or_path=${main_model_name_or_path} \
#        data_args.main_model_step_idx=${step_idx} \
#        data_args.wo_cot=${wo_cot} \
#        data_args.dataset_name=EleutherAI/hendrycks_math \
#        data_args.max_predict_samples=100
#        data_args.dataset_subset_name=${subset_name} \
#      echo "Finished running classmate ${model_name_or_path} on hendrycks math inference on ${main_model_name_or_path} step  ${step_idx} on subset: ${subset_name}"
#      done
#
#      # AIME
#      CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/main_classmate_utility.py \
#          model_args.model_name_or_path=${model_name_or_path} \
#          data_args.main_model_name_or_path=${main_model_name_or_path} \
#          data_args.main_model_step_idx=${step_idx} \
#          data_args.wo_cot=${wo_cot} \
#          data_args.dataset_name=AI-MO/aimo-validation-aime \
#          data_args.dataset_split_name=train \
#          data_args.dataset_subset_name=default
#      echo "Finished running classmate ${model_name_or_path} on AIME inference main model at step ${step_idx}"
      echo "Finished classmate ${model_name_or_path} on ${main_model_name_or_path} step ${step_idx}"
    done
  )
done

#kill all process:
#ps -u "$USER" -f | grep -- "python ./src/main.py --base_configs=./configs/cot_utility_to_classmate.yaml" | grep -v grep | awk '{print $2}' | xargs -r kill

#bash scripts/cot_utility_to_classmate.sh





## split 0-9 across 4 gpus: 0-1, 2-3, 4-6, 7-9
#multiple_of=1
#step_size=31
#gpu_idx=0
#start_idx=0
#max_idx=1
##start_idx=2
##max_idx=3
##start_idx=4
##max_idx=6
##start_idx=7
##max_idx=9
#for (( i=start_idx; i<=max_idx; i+=multiple_of )); do
#  # Launch a parallel job for each j in the current group [i, i+multiple_of-1], clipped to max_idx
#  for (( j=i; j<i+multiple_of && j<=max_idx; j++ )); do
#    if [[ "${j}" == "wo_cot" ]]; then
#      wo_cot=True
#      step_idx="wo_cot"
#    elif [[ "${j}" == "base" ]]; then   #allenai/OLMo-2-0425-1B-DPO
#      step_idx="base"
#      wo_cot=False
#    else
#      wo_cot=False
#      step_idx=$((step_size * (j + 1)))
#    fi
#
#    (
#      for model_name_or_path in \
#        "meta-llama/Llama-3.2-3B-Instruct" \
#        "allenai/OLMo-2-0425-1B-Instruct" \
#        "Qwen/Qwen2.5-7B-Instruct"; do
##        "meta-llama/Llama-3.1-8B-Instruct"; do
#
#        echo "[j=${j}] Start running classmate ${model_name_or_path} on main_model_step_${step_idx} (wo_cot=${wo_cot})"
#        CUDA_VISIBLE_DEVICES="${gpu_idx}" python "${SRC_DIR}/main.py" \
#          --base_configs="${CONFIG_DIR}/${base_config}" \
#          --main_model_step_idx "${step_idx}" \
#          --model_name_or_path "${model_name_or_path}" \
#          --wo_cot "${wo_cot}"
#        echo "[j=${j}] Finished classmate ${model_name_or_path} on main_model_step_${step_idx}"
#      done
#    ) &
#  done
#
#  # Wait for all jobs in this group to finish before moving to the next
#  wait
#  echo "== Finished group starting at ${i} =="
#done

#kill all process:
#ps -u "$USER" -f | grep -- "python ./src/main.py --base_configs=./configs/cot_utility_to_classmate.yaml"" | grep -v grep | awk '{print $2}' | xargs -r kill
#
#bash scripts/cot_utility_to_classmate.sh