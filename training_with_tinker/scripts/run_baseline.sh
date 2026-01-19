
export TINKER_API_KEY=tml-qNTmad59cflqHptwnHs7mS6fcJhqB6oj2bvaJdHpNzxHem4ctaeMSHhmqImBNtKhOAAAA
export PYTHONPATH=:${PYTHONPATH}

dataset_name=hendrycks_math_minimal_answer_box_prompt_105_test
model_name_or_path=Qwen/Qwen3-8B-Base
main_cot_keep_rate=0.5

adv_estimator=grpo_baseline
#adv_estimator=grpo_classmate
#adv_estimator=grpo_main_classmate_separated



python training_with_tinker/main.py \
  training_args.adv_estimator=$adv_estimator \
  data_args.dataset_name=$dataset_name \
  main_model_args.model_name_or_path=$model_name_or_path \
  training_args.wandb_run_name="${adv_estimator}-${model_name_or_path//\//_}-${dataset_name}_keep_m_${main_cot_keep_rate}" \
  classmate_model_args.main_cot_keep_rate=$main_cot_keep_rate

#bash training_with_tinker/scripts/run_classmate.sh