
export TINKER_API_KEY=tml-qNTmad59cflqHptwnHs7mS6fcJhqB6oj2bvaJdHpNzxHem4ctaeMSHhmqImBNtKhOAAAA
export PYTHONPATH=:${PYTHONPATH}

#dataset_name=mmlu_sycophancy
dataset_name=mmlu_q_hint_sep
model_name_or_path=Qwen/Qwen3-8B-Base
#model_name_or_path=Qwen/Qwen3-4B-Instruct-2507

classmate_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct

eval_main_cot_keep_rate=0.5

adv_estimator=grpo_baseline
#adv_estimator=grpo_classmate
#adv_estimator=grpo_main_classmate_separated



python training_with_tinker/main.py \
  training_args.adv_estimator=$adv_estimator \
  data_args.dataset_name=$dataset_name \
  main_model_args.model_name_or_path=$model_name_or_path \
  training_args.wandb_run_name="${dataset_name}/${adv_estimator}-${model_name_or_path//\//_}-cl_${classmate_model_name_or_path//\//_}_eval_keep_m_${eval_main_cot_keep_rate}" \
  classmate_model_args.model_name_or_path=${classmate_model_name_or_path} \
  classmate_model_args.eval_main_cot_keep_rate=${eval_main_cot_keep_rate}

#bash training_with_tinker/scripts/mmlu_run_baseline.sh
#conda deactivate
#conda activate tinker