conda install setuptools

git clone https://github.com/thinking-machines-lab/tinker-cookbook.git

(
  cd tinker-cookbook
  pip install -e .
)

pip install -r my_requirements.txt

# Install flash attention and flash infer
# Change CUDA, torch, abiTRUE/FALSE, python version in the following line as needed
#wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl && pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install --no-deps --no-cache-dir flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
#wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl && pip install --no-cache-dir flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install flashinfer-python
pip install lm_eval
pip install lm-eval[math]

bash my_scripts/build_omegaconf_hydra_w_antlr_4110.sh

python -c "import nltk; nltk.download('punkt_tab')"