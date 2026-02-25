

## Setup server environment

### Install zsh and Oh My Zsh
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

### Install docker

[//]: # (TODO need to complete)
- Follow the official Docker installation guide for your operating system: https://docs.docker.com/get-docker/
- After installation, verify that Docker is installed correctly by running:
  ```bash
  docker --version
  ```

### Create user
- create user:
  - adduser lorena
  - Generate ssh key:
    ```bash
    ssh-keygen -t ed25519 -f $HOME/.ssh/$USER -N ""
    chown -R $USER:$USER $HOME/.ssh
    chmod 700 $HOME/.ssh
    chmod 600 $HOME/.ssh/$USER
    chmod 644 $HOME/.ssh/$USER.pub
    cat $HOME/.ssh/$USER.pub >> $HOME/.ssh/authorized_keys
    chmod 600 $HOME/.ssh/authorized_keys
    ```

## Install environment for the project
### Install docker:
- Install dependency with docker:
  - If encounter layer registration permission issue, then need to change the data-root-dir to a non-nsf filesystem.
    ~/.config/docker/daemon.json
    ```
    {
    "/local/data/lorena/docker-storage"
     }
    systemctl --user restart docker
    ```
- Build docker image:
  ```
  bash my_scripts/create_docker.sh
  docker start verl
  docker exec -it verl bash
  ```
- Rebuild docker container
  ```bash
  docker stop verl
  docker rm verl
  bash my_scripts/create_docker.sh
  docker start verl
  docker exec -it verl bash
  ```
- Install zsh/oh-my-zsh in docker container:
  ```
  apt update
  apt install zsh
  zsh --version
  sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
  ```
- Change ~/.bashrc:
  ```
  cd verl
  zsh
  ```
- 

### Install with miniconda (if not docker):
- ```
  curl -LsSf https://astral.sh/uv/0.9.14/install.sh | sh
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
  source ~/.bashrc
  ```
- Setup miniconda:
  ```
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
  bash miniconda.sh -b -p $HOME/midwest/miniconda
  ```
  - ```vim ~/.bashrc```:
    ```
    export PATH="$HOME/miniconda/bin:$PATH"
    ```
    - For my lambda gpu:
      ``` 
      export PATH="$HOME/east/miniconda/bin:$PATH"
      export PATH="$HOME/midwest/miniconda/bin:$PATH"
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
  - Modify bashrc:
    ```
    conda activate verl_tinker
    cd $HOME/midwest/classmate-cot-verl
    cd $HOME/east/classmate-cot-verl
    ```
[//]: # (  conda create -n verl python==3.10.12)
- Install dependency:
  ```
  conda create -n verl_tinker python==3.11
  conda activate verl_tinker
  bash install.sh
  ```
- Login to HF and wandb:
  ```
  hf auth login
  wandb login
  ```
[//]: # (  USE_MEGATRON=0 USE_SGLANG=0 bash scripts/install_vllm_sglang_mcore.sh)
[//]: # (  pip install --no-deps -e .)
  - Note:
    - Currently not supporting megatron and sglang
    - [build_omegaconf_hydra_w_antlr_4111.sh](scripts/build_omegaconf_hydra_w_antlr_4110.sh) is called in install_vllm_sglang_mcore.sh to build compatible omegaconf and hydra versions with antlr 4.11.1 for using parse_latex for hendrycks math

# Modify for classmate CoT training
- Dataset-specific preprocessing and reward
  - Preprocessing scripts:
    - [data_preprocessing](verl/data_preprocessing)
    - [prepare_data.sh](my_scripts/prepare_data.sh): ```bash my_scripts/prepare_data.sh```
  - Common utils: [math_utils.py](verl/utils/math_utils.py)
- Add config: [classmate_cot_ppo_trainer.yaml](verl/trainer/config/classmate_cot_ppo_trainer.yaml)
  - (optional) custom compute_score function
- [classmate training main ppo script](verl/trainer/deprecate_classmate_cot_main_ppo.py):
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
      - ```_prepare_classmate_batch()```: take CoT token ids from the main model and reformat them for classmate; change prompts based on reward_type / continue_mode
      - Use [ClassmateWorker](verl/workers/classmate_workers.py) to do the sampling
        - Take in model_path and initialize model and tokenizer
        - Return outputs with shape (bsz, classmate_num, num_return_sequences)
        - ```_generate_classmate_continuations()```: 
    - ```marked_timer("reward", timing_raw, color="yellow"):```: 
      Use [classmate reward manager](verl/workers/reward_manager/classmate_cot_rm.py) to calculate classmate rewards (built on top of naive reward manager for GRPO)
      - Calculate Classmate Rewards: For each batch item, iterate through all samples from each classmate model, calculate each sample's reward one by one, average across samples for each classmate, and do weighted average across all classmate models (TODO: now the classmate's weights are fixed to be average)
- Dataset-specific rewards:
  - Dolci:
    - [olmo_verifiers.py](verl/utils/reward_score/olmo_verifiers.py): verifiers + ```def compute_score```
    - [olmo3_judge_utils.py](verl/utils/reward_score/olmo3_judge_utils.py)
  - Old setting:
    - [__init__.py](verl/utils/reward_score/__init__.py): def default_compute_score
    - [gsm8k.py](verl/utils/reward_score/gsm8k.py)
    - [hendrycks_math.py](verl/utils/reward_score/hendrycks_math.py)
    - [code_contests_modify_code.py](verl/utils/reward_score/code_contests_modify_code.py)
- Printing training metrics: [metric_utils.py](verl/trainer/ppo/metric_utils.py) -> def compute_data_metrics
- Train on coding task:
  - Sandbox_fusion: [__init__.py](verl/utils/reward_score/sandbox_fusion/__init__.py): pass in the programming language
  - Pass in sandbox fusion url: ppo_trainer.yaml config files, ppo_trainer.py, reward_manager (naive.py, classmate_cot_rm.py)
  - compute_score: check the ```elif code_contests_modify_code``` in ```default_compute_score```

# Code Structure (after refactoring for sycophancy)
- Configs:
  - Baseline: [qwen_ppo_trainer.yaml](verl/trainer/config/qwen_ppo_trainer.yaml)
  - With classmate reward: [qwen_classmate_cot_ppo_trainer.yaml](verl/trainer/config/qwen_classmate_cot_ppo_trainer.yaml)
- Training entry:
  - Baseline: [qwen_main_ppo.py](verl/trainer/qwen_main_ppo.py) 
  - With classmate reward: [qwen_classmate_cot_main_ppo.py](verl/trainer/qwen_classmate_cot_main_ppo.py)
- Trainers:
  - Baseline: [baseline_ray_trainer.py](verl/trainer/ppo/baseline_ray_trainer.py)
  - With classmate: [classmate_cot_ray_trainer.py](verl/trainer/ppo/classmate_cot_ray_trainer.py)
  - Monitoring during validation is handled in the Trainer files with ```monitor_cot_wrapper_w_tinker```
- Calculate reward:
  - Reward manager: 
    - [sycophancy_classmate_cot_rm.py](verl/workers/reward_manager/sycophancy_classmate_cot_rm.py)
    - Register rm:
      - ```@register("sycophancy_classmate_cot")``` on top of the rm class
      - Update [__init__.py](verl/workers/reward_manager/__init__.py)
      - Change ```reward_manager``` in training config: [qwen_classmate_cot_ppo_trainer.yaml](verl/trainer/config/qwen_classmate_cot_ppo_trainer.yaml)
    - Handle making llm judge api calls with ```send_prompt_to_tinker```
  - Reward function/verifier:
    - Base class you should inherit: [BaseVerifier.py](verl/utils/reward_score/cot_monitor/BaseVerifier.py)
      - compute_metrics report binary/non-binary metrics
      - get_verifier: return specific verifier based on the given data source
    - When you do not have gt and need llm judge:
      - Judge prompt and output parsings are handled by specific verifier you implemented (that inherits the base class)
    - When you have gt: ```compute_score``` is from [BaseVerifier.py](verl/utils/reward_score/cot_monitor/BaseVerifier.py)
- What to do when adding new tasks and verifiers:
  - Data preprocessing script:
    - Sycophancy: [h4_helpful_instructions.py](verl/data_preprocessing/h4_helpful_instructions.py)
  - Implement BaseVerifier subclass
  - Update ```get_verifier```

# Hosting Sandbox for coding tasks
- Dolci code/code_stdio:
  - ```
    uvicorn verl.code_utils.api:app --host 0.0.0.0 --port 8001
    ```
  - Test connection:
    ```
    curl -X GET http://localhost:8001/health
    curl -X POST http://localhost:8001/test_program -H "Content-Type: application/json" -d '{"program": "def add(a, b): return a + b", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"], "max_execution_time": 1.0}'
    curl -X POST http://localhost:8001/test_program_stdio -H "Content-Type: application/json" -d '{"program": "import sys\nfor line in sys.stdin.read().splitlines():\n    print(int(line.strip()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}, {"input": "100\n", "output": "101\n"}], "max_execution_time": 1.0}'
    ```
- Host [SandBox Fusion](https://bytedance.github.io/SandboxFusion/docs/docs/get-started#local-deployment) (create a sandbox to test code safely):
  - Note: need a machine with root access
  - ```docker run -it -p YOUR_PORT_NUMBER:8080 volcengine/sandbox-fusion:server-20250609``` (Uvicorn always starts on port 8080 in docker. Map it to some port number on your machine)
  - Test connection:
    ```
    curl 'http://localhost:YOUR_PORT_NUMBER/run_code' -H 'Content-Type: application/json'   --data-raw '{"code": "print(\"Hello, world!\")", "language": "python"}'
    ```

- Some painpoints:
  - Took some times to navigate the entire pipeline (it's fairly modular, but takes some time to figure out how different actors interact & got scheduled. Should have read their docs first)
  - How to initialize workers and share resource pool with other actor & reference rollout workers
  - How to do classmate sampling in parallel + onload & offload models before and after sampling
  - Just hard to debug: can't use pdb in ray actors. Need to wait for a while to start the program every time.
  - How to create multiple vLLM engines (still unsolved)


# Run Experiments
## Preprocess datasets
- Dataset-specific scripts are in [data_preprocessing](verl/data_preprocessing):
  - The scripts extract clean solutions from the given solution CoTs and concat question with instruction following templates.
  - Change things like output directories if needed.
- Run [prepare_data.sh](my_scripts/prepare_data.sh):
  ```
  bash my_scripts/prepare_data.sh
  ```

## Training
### Baseline
- gsm8k:
  ```
  bash my_scripts/run_olmo_baseline.sh
  ```
- hendrycks_math:
  ```
  bash my_scripts/run_olmo_hendrycks_baseline.sh
  ```
  
### Train with classmate
- Host and test remote classmate vllm engines:
  - Directly run vllm serve: [host_classmate_model.sh](my_scripts/host_classmate_model.sh)
  - Then change port number in outputs/host_classmate_models/classmate_model_mapping.json
- gsm8k:
  ```
  bash my_scripts/run_olmo_classmate_cot.sh
  ```
- hendrycks_math:
  ```
  bash my_scripts/run_olmo_hendrycks_classmate_cot.sh
  ```
- Merge ckpts: [merge_fsdp_ckpts.sh](my_scripts/merge_fsdp_ckpts.sh)

# Others
- Storage path:
  - /local/data/lorena/
  - /proj/interaction/interaction-filer/lorena/classmate-cot-verl
- Hosting classmate on lambda: