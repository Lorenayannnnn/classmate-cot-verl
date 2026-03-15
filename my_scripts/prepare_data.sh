
export PYTHONPATH=:${PYTHONPATH}

echo "Preparing general reward dataset for training..."
python3 verl/data_preprocessing/general_reward.py --seed 0
python3 verl/data_preprocessing/general_reward.py --seed 1
python3 verl/data_preprocessing/general_reward.py --seed 2

echo "Preparing sycophancy dataset for training..."
python3 verl/data_preprocessing/sycophancy.py  --seed 0
python3 verl/data_preprocessing/sycophancy.py  --seed 1
python3 verl/data_preprocessing/sycophancy.py  --seed 2

echo "Preparing confidence dataset for training..."
python3 verl/data_preprocessing/confidence.py  --seed 0
python3 verl/data_preprocessing/confidence.py  --seed 1
python3 verl/data_preprocessing/confidence.py  --seed 2

echo "Preparing longer response dataset for training..."
python3 verl/data_preprocessing/longer_response.py  --seed 0
python3 verl/data_preprocessing/longer_response.py  --seed 1
python3 verl/data_preprocessing/longer_response.py  --seed 2

echo "Preparing unsafe compliance dataset for training..."
python3 verl/data_preprocessing/unsafe_compliance.py  --seed 0
python3 verl/data_preprocessing/unsafe_compliance.py  --seed 1
python3 verl/data_preprocessing/unsafe_compliance.py  --seed 2


#bash my_scripts/prepare_data.sh