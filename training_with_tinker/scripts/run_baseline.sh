
export TINKER_API_KEY=tml-qNTmad59cflqHptwnHs7mS6fcJhqB6oj2bvaJdHpNzxHem4ctaeMSHhmqImBNtKhOAAAA
export PYTHONPATH=:${PYTHONPATH}

dataset_name=hendrycks_math_minimal_answer_box_prompt_105_test
#model_name_or_path=Qwen/Qwen3-8B-Base
model_name_or_path=Qwen/Qwen3-4B-Instruct-2507
eval_main_cot_keep_rate=0.7

adv_estimator=grpo_baseline
#adv_estimator=grpo_classmate
#adv_estimator=grpo_main_classmate_separated



python training_with_tinker/main.py \
  training_args.adv_estimator=$adv_estimator \
  data_args.dataset_name=$dataset_name \
  main_model_args.model_name_or_path=$model_name_or_path \
  training_args.wandb_run_name="${adv_estimator}-${model_name_or_path//\//_}-${dataset_name}_eval_keep_m_${eval_main_cot_keep_rate}" \
  classmate_model_args.eval_main_cot_keep_rate=${eval_main_cot_keep_rate}

#bash training_with_tinker/scripts/run_baseline.sh
#conda deactivate
#conda activate tinker