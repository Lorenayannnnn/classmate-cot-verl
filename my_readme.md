

## Setup environment
- Install zsh:
  ```
  sudo apt update
  sudo apt install zsh

  zsh --version
  ```
- Install Oh My Zsh (optional but recommended):
  ```bash
  sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
  ```
  This will install Oh My Zsh and automatically backup your existing `.zshrc` file.


- create user:
  - adduser lorena
  - Generate ssh key:
    ```bash
    ssh-keygen -t ed25519 -f /home/lorena/.ssh/lambda_lorena -N ""
    chown -R lorena:lorena /home/lorena/.ssh
    chmod 700 /home/lorena/.ssh
    chmod 600 /home/lorena/.ssh/lambda_lorena
    chmod 644 /home/lorena/.ssh/lambda_lorena.pub
    cat /home/lorena/.ssh/lambda_lorena.pub >> /home/lorena/.ssh/authorized_keys
    chmod 600 /home/lorena/.ssh/authorized_keys
    ```

- Setup miniconda:
  ```
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  ```
  - to ~/.bashrc / ~/.zshrc, add:
    ```
    export PATH="$HOME/miniconda/bin:$PATH"
    ```
  - restart terminal or run:
    ```
    source ~/.bashrc
    ```
  - ```conda init```
  - Verify installation:
    ```
    conda --version
    ```

- Install dependency:
  - conda create -n verl python==3.12
  - conda activate verl
  - bash scripts/install_vllm_sglang_mcore.sh
  - pip install --no-deps -e .


## Modify for classmate CoT

- Add config: [classmate_cot_ppo_trainer.yaml](verl/trainer/config/classmate_cot_ppo_trainer.yaml)
  - reward_model:
    - reward_manager: classmate_cot     # verl/workers/reward_manager/classmate_cot.py

    - classmate_cot_reward_configs:
      # "Qwen/Qwen2.5-7B-Instruct", "mistralai/Ministral-8B-Instruct-2410", "meta-llama/Llama-3.1-8B-Instruct"
      # "meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "allenai/OLMo-2-0425-1B-Instruct", "allenai/OLMo-2-0425-1B-SFT"
      classmate_model_name_or_path_list: ["meta-llama/Llama-3.2-1B-Instruct"]

      classmate_reward_weight: 1.0

      # vanilla, remove_wo_cot, random_truncate, random_truncate_remove_wo_cot, random_truncate_step_wise_utility
      classmate_reward_type: vanilla_reward

      # "immediate_answer", "continue_cot"
      classmate_continue_mode: continue_cot
    
    - classmate_generation_configs:
      max_new_tokens: 256
      do_sample: True
      temperature: 0.7
      top_p: 0.9
      num_return_sequences: 1

  - (optional) custom compute_score function
- [classmate training main ppo script](verl/trainer/classmate_cot_main_ppo.py):
  - Change config file name at the top
  - add_classmate_rollout_worker() and call it in run() in TaskRunner
  - Change reward manager & ClassmateCoTRayPPOTrainer loading
- Add [ClassmateCoTRayPPOTrainer](verl/trainer/ppo/classmate_cot_ray_trainer.py):
  - __init__(): change init to take in classmate reward model configs
  - def init_workers():
    - Create resource pool for ClassmateWorker
    - create_classmate_workers()
  - Update ```fit()```:
    - ```batch = self._generate_classmate_continuations(batch)```
      - ```_prepare_classmate_batch()```: change prompts based on reward_type / continue_mode
      - Use [ClassmateWorker](verl/workers/classmate_workers.py) to do the sampling
        - Take in model_path and initialize model and tokenizer
        - Return outputs with shape (bsz, classmate_num, num_return_sequences)
        - ```_generate_classmate_continuations()```: 
    - ```marked_timer("reward", timing_raw, color="yellow"):```: 
      Use [classmate reward manager](verl/workers/reward_manager/classmate_cot_rm.py) to calculate classmate rewards (built on top of naive reward manager for GRPO)
      - Calculate Classmate Rewards: For each batch item, iterate through all samples from each classmate model, calculate each sample's reward one by one, average across samples for each classmate, and do weighted average across all classmate models (TODO: now the classmate's weights are fixed to be average)


- Some painpoints:
  - Took some times to navigate the entire pipeline (it's fairly modular, but takes some time to figure out how different actors interact & got scheduled. Should have read their docs first)
  - How to initialize workers and share resource pool with other actor & reference rollout workers
  - How to do classmate sampling in parallel + onload & offload models before and after sampling
  - Just hard to debug: can't use pdb in ray actors. Need to wait for a while to start the program every time.
  - How to create multiple vLLM engines (still unsolved)