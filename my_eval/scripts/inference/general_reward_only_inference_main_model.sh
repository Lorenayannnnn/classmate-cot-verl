SRC_DIR=./my_eval/src
export VLLM_WORKER_MULTIPROC_METHOD=spawn

#This exits the script if any command fails
set -e

export PYTHONPATH=:${PYTHONPATH}

main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_all_tokens-seed_0
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_all_tokens-seed_1
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_all_tokens-seed_2
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_cot_only-seed_0
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_cot_only-seed_1
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-baseline_cot_only-seed_2
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_self-seed_0
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_self-seed_1
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_self-seed_2
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_llama-seed_0
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_llama-seed_1
#main_model_name_or_path=LorenaYannnnn/general_reward-Qwen3-0.6B-OURS_llama-seed_2

base_model_name_or_path=Qwen/Qwen3-0.6B
think_start_str="<think>"
think_end_str="</think>"

main_cot_keep_rate=1

gpu_idx=0

step_size=40           # checkpoint save frequency
max_training_steps=920 # total RL training steps (update to match actual run)
num_eval_ckpts=6
last_ckpt=$(( (max_training_steps / step_size) * step_size ))
viz_step_size=$(( (last_ckpt / num_eval_ckpts / step_size) * step_size ))
viz_offset=$(( last_ckpt - viz_step_size * num_eval_ckpts ))

run_base=false  # set to true to run base model checkpoint once before RL steps
dataset_split_name=test

monitor_model_name=null
monitor_backend_type=null
llm_judge_model_name=null
llm_judge_backend_type=null
general_reward_model_name=null
general_reward_backend_type=hf_scoring

_run_inference() {
  local step_idx=$1
  echo "Inference main model at step ${step_idx}"
  CUDA_VISIBLE_DEVICES=${gpu_idx} python ${SRC_DIR}/inference_main_model.py \
      model_args.model_name_or_path=${main_model_name_or_path} \
      model_args.base_model_name_or_path=${base_model_name_or_path} \
      model_args.main_model_step_idx=${step_idx} \
      model_args.main_cot_keep_rate=${main_cot_keep_rate} \
      running_args.generation_args.max_tokens=3072 \
      running_args.monitor_model_name=${monitor_model_name} \
      running_args.monitor_backend_type=${monitor_backend_type} \
      running_args.llm_judge_model_name=${llm_judge_model_name} \
      running_args.llm_judge_backend_type=${llm_judge_backend_type} \
      running_args.general_reward_model_name=${general_reward_model_name} \
      running_args.general_reward_backend_type=${general_reward_backend_type} \
      data_args.dataset_split_name=${dataset_split_name} \
      data_args.dataset_name=general_reward \
      data_args.max_predict_samples=500 \
      "model_args.think_start_str='${think_start_str}'" \
      "model_args.think_end_str='${think_end_str}'"
  echo "Finished inference main model at step ${step_idx}"
}

if ${run_base}; then
  ( _run_inference "base" )
fi

for i in $(seq 1 ${num_eval_ckpts}); do
  step_idx=$((viz_offset + viz_step_size * i))
  ( _run_inference "${step_idx}" )
done

#bash my_eval/scripts/inference/general_reward_only_inference_main_model.sh
