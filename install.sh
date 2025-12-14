pip install -r my_requirements.txt

# Install flash attention and flash infer
# Change CUDA, torch, abiTRUE/FALSE, python version in the following line as needed
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl && pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install flashinfer-python==0.5.3
pip install lm_eval==0.4.9.2
pip install lm-eval[math]

bash scripts/build_omegaconf_hydra_w_antlr_4110.sh

python -c "import nltk; nltk.download('punkt_tab')"