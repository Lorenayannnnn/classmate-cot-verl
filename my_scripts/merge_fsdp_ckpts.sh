
# max length 96
main_dir="/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs/general_reward_Qwen/Qwen3-0.6B_grpo_baseline_192000_episodes_seed_42"
repo_name="20260314-Skywork_qwen_0.6B-Qwen3-0.6B_grpo_baseline_192000_episodes_seed_42"

#main_dir="/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs/general_reward_Qwen/Qwen3-0.6B_grpo_w_classmate_cl_192000_episodes_seed_42"
#repo_name="20260314-Skywork_qwen_0.6B-Qwen3-0.6B_grpo_w_classmate_cl_192000_episodes_seed_42"

#main_dir="/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs/general_reward_Qwen/Qwen3-0.6B_grpo_baseline_cot_only_192000_episodes_seed_42"
#repo_name="20260314-Skywork_qwen_0.6B-Qwen3-0.6B_grpo_baseline_cot_only_192000_episodes_seed_42"

#main_dir="/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs/general_reward_Qwen/Qwen3-0.6B_grpo_baseline_output_only_192000_episodes_seed_42"
#repo_name="20260314-Skywork_qwen_0.6B-Qwen3-0.6B_grpo_baseline_output_only_192000_episodes_seed_42"




start_idx=0
max_idx=9
step_size=40

for (( i=start_idx; i<=max_idx; i++ )); do
  echo "== Starting group beginning at index ${i} =="
  # Launch a parallel job for each j in the current group [i, i+multiple_of-1], clipped to max_idx
  for (( j=i; j<i+1 && j<=max_idx; j++ )); do
    # Example step mapping; adjust as needed
    step_idx=$((step_size * (j + 1)))
    (
      echo "[j=${j}] Merge FSDP at step ${step_idx}"
      python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir ${main_dir}/global_step_${step_idx}/actor \
        --target_dir ${main_dir}/global_step_${step_idx}/actor/huggingface
      echo "[j=${j}] Merge FSDP at step ${step_idx}"
    )
  done
  # Wait for all jobs in this group to finish before moving to the next group
  echo "== Finished group starting at ${i} =="
done

python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir ${main_dir}/global_step_430/actor \
  --target_dir ${main_dir}/global_step_430/actor/huggingface


python upload_ckpts_to_huggingface.py \
  --root_path ${main_dir} \
  --repo_name ${repo_name} \
#  --main_only_path /local/data/lorena/classmate_cot_w_verl/outputs/helpful_instructions_Qwen/Qwen3-0.6B_grpo_warmup_24000_episodes_seed_42/global_step_12

#bash my_scripts/merge_fsdp_ckpts.sh