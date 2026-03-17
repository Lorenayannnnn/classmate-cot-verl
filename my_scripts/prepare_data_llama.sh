
export PYTHONPATH=:${PYTHONPATH}

echo "Preparing general reward dataset (llama format) for training..."
python3 verl/data_preprocessing/llama/general_reward.py --seed 0
python3 verl/data_preprocessing/llama/general_reward.py --seed 1
python3 verl/data_preprocessing/llama/general_reward.py --seed 2

#bash my_scripts/prepare_data_llama.sh