
export PYTHONPATH=:${PYTHONPATH}

#python3 verl/data_preprocessing/*.py
#python3 verl/data_preprocessing/gsm8k.py
python3 verl/data_preprocessing/hendrycks_math.py
#python3 verl/data_preprocessing/leetcode.py
#python3 verl/data_preprocessing/dolci.py
#python3 verl/data_preprocessing/olmo2_mix.py \
#python3 verl/data_preprocessing/DeepScaleR.py \
#  --model_name_or_path allenai/OLMo-2-1124-7B-DPO
#python3 verl/data_preprocessing/gsm_math.py \



#bash my_scripts/prepare_data.sh