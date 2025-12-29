


#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes"
#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_allenai/OLMo-2-1124-7B-DPO_RLVR-GSM-MATH-IF-Mixed-Constraints_baseline_237392_episodes"
#main_dir="/local/data/lorena/classmate_cot_w_verl/outputs/grpo_allenai/OLMo-2-1124-7B-DPO_RLVR-GSM-MATH-IF-Mixed-Constraints_with_classmate_reward_llama_237400_episodes"
#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-4B-Base_DeepScaleR_baseline_322512_episodes"
#main_dir="/local/data/lorena/classmate_cot_w_verl/outputs/grpo_Qwen/Qwen3-4B-Base_DeepScaleR_w_classmate_llama_322512_episodes"
#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-4B-Base_DeepScaleR_w_classmate_llama0.5_322512_episodes"
#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-0.6B_GSM_MATH_baseline_357024_episodes"
#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-1.7B_GSM_MATH_baseline_357024_episodes"
#main_dir="/local/data/lorena/classmate_cot_w_verl/outputs/grpo_Qwen/Qwen3-1.7B_GSM_MATH_w_1_classmate_llama_357024_episodes"
#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-0.6B_GSM_MATH_w_1_classmate_llama_357024_episodes_seed_1"
#main_dir="/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-0.6B_hendrycks_math_think_step_by_step_baseline_359664_episodes_seed_42"
#main_dir="/local/data/lorena/classmate_cot_w_verl/outputs/grpo_Qwen/Qwen3-0.6B_gsm8k_think_prompt_w_1_classmate_llama_358704_episodes_seed_42"
main_dir="/local/data/lorena/classmate_cot_w_verl/outputs/grpo_Qwen/Qwen3-0.6B_gsm8k_minimal_answer_box_prompt_w_1_classmate_llama_717408_episodes_seed_42"

repo_name="20251228-Qwen3-0.6B_gsm8k_minimal_answer_box_w_1_classmate_llama_717408_episodes_seed_42"

#start_idx=0
#max_idx=13
#step_size=12
#for (( i=start_idx; i<=max_idx; i++ )); do
#  echo "== Starting group beginning at index ${i} =="
#  # Launch a parallel job for each j in the current group [i, i+multiple_of-1], clipped to max_idx
#  for (( j=i; j<i+1 && j<=max_idx; j++ )); do
#    # Example step mapping; adjust as needed
#    step_idx=$((step_size * (j + 1)))
#    (
#      echo "[j=${j}] Merge FSDP at step ${step_idx}"
#      python -m verl.model_merger merge \
#        --backend fsdp \
#        --local_dir ${main_dir}/global_step_${step_idx}/actor \
#        --target_dir ${main_dir}/global_step_${step_idx}/actor/huggingface
#      echo "[j=${j}] Merge FSDP at step ${step_idx}"
#    )
#  done
#  # Wait for all jobs in this group to finish before moving to the next group
#  echo "== Finished group starting at ${i} =="
#done
#python -m verl.model_merger merge \
#  --backend fsdp \
#  --local_dir ${main_dir}/global_step_174/actor \
#  --target_dir ${main_dir}/global_step_174/actor/huggingface

#python -m verl.model_merger merge \
#  --backend fsdp \
#  --local_dir ${main_dir}/global_step_175/actor \
#  --target_dir ${main_dir}/global_step_175/actor/huggingface

python upload_ckpts_to_huggingface.py \
  --root_path ${main_dir} \
  --repo_name ${repo_name}

#bash my_scripts/merge_fsdp_ckpts.sh