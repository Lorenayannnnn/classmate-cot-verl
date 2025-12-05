
export PYTHONPATH=:${PYTHONPATH}

#python3 verl/data_preprocessing/*.py
#python3 verl/data_preprocessing/gsm8k.py
#python3 verl/data_preprocessing/hendrycks_math.py
#python3 verl/data_preprocessing/leetcode.py

python3 verl/data_preprocessing/dolci.py \
  --model_name_or_path allenai/Olmo-3-7B-Think-DPO \
  --max_prompt_length 1024 \
  --max_train_sample 25500  # 25% of the full training set


#bash my_scripts/prepare_data.sh