# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a fork of [verl](https://github.com/volcengine/verl) (Volcano Engine RL for LLMs) extended with **Classmate Chain-of-Thought (CoT)** training — a method where a "classmate" model observes the main model's CoT reasoning and samples continuations, providing a correction signal to reduce sycophancy in LLM outputs.

## Environment Setup

**Recommended: conda + install script**
```bash
conda create -n verl_tinker python==3.11
conda activate verl_tinker
bash install.sh
hf auth login
wandb login
```

**Alternative: Docker**
```bash
bash my_scripts/create_docker.sh
docker start verl
docker exec -it verl bash
```

Note: Megatron and SGLang are not currently supported.

## Common Commands

### Preprocess datasets
```bash
bash my_scripts/prepare_data.sh
```

### Training

**Warmup stage** (GRPO on helpful_instructions before sycophancy training):
```bash
bash my_scripts/qwen3_warmup_sycophancy.sh
```

**Baseline training** (GRPO, no classmate):
```bash
bash my_scripts/qwen3_baseline_general_sycophancy.sh
```

**Classmate CoT training** (GDPO with classmate model):
```bash
bash my_scripts/qwen3_classmate_general_sycophancy.sh
```

**Self-play variant** (classmate = main model):
```bash
bash my_scripts/self_qwen3_classmate_general_sycophancy.sh
```

Entry points:
- Baseline: `python3 -m verl.trainer.qwen_main_ppo`
- Classmate: `python3 -m verl.trainer.qwen_classmate_cot_main_ppo`

### Merge FSDP checkpoints
```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir <checkpoint_dir>/global_step_N/actor \
  --target_dir <checkpoint_dir>/global_step_N/actor/huggingface
```
Or use `my_scripts/merge_fsdp_ckpts.sh` (edit `main_dir` and `repo_name` at top of file).

### Upload checkpoints to HuggingFace
```bash
python upload_ckpts_to_huggingface.py --root_path <main_dir> --repo_name <repo_name>
```

### Host classmate model as vLLM server
```bash
bash my_scripts/host_classmate_model.sh
# Then update outputs/host_classmate_models/classmate_model_mapping.json with port numbers
```

### Host local sandbox for coding tasks
```bash
uvicorn verl.code_utils.api:app --host 0.0.0.0 --port 8001
```

## Architecture

### Training Pipeline

```
Data → Rollout (vLLM) → [Classmate Sampling] → Reward → Advantage → PPO/GRPO Update
```

1. **Rollout**: Main model generates full responses (with CoT) via vLLM
2. **Classmate sampling** *(classmate-only)*: `ClassmateWorker` takes the main model's CoT prefix and samples continuations using a separate smaller model (e.g., Qwen3-0.6B, Llama-3.2-1B)
3. **Reward**: `SycophancyClassmateCoTRewardManager` computes base reward + classmate correction signal; combined via weighted average
4. **Advantage**: GAE or GRPO
5. **Policy update**: FSDP-based actor update

### Key Custom Files

| File | Role |
|------|------|
| `verl/trainer/qwen_classmate_cot_main_ppo.py` | Main entry point for classmate training |
| `verl/trainer/qwen_main_ppo.py` | Main entry point for baseline training |
| `verl/trainer/ppo/classmate_cot_ray_trainer.py` | `ClassmateCoTRayPPOTrainer` — extends base trainer with classmate sampling/rewards |
| `verl/trainer/ppo/baseline_ray_trainer.py` | Base PPO/GRPO Ray trainer |
| `verl/workers/classmate_workers.py` | `ClassmateWorker` — handles classmate model inference via vLLM |
| `verl/workers/reward_manager/sycophancy_classmate_cot_rm.py` | `SycophancyClassmateCoTRewardManager` — registered as `"sycophancy_classmate_cot"` |
| `verl/trainer/config/qwen_classmate_cot_ppo_trainer.yaml` | Primary config for classmate training |
| `verl/utils/reward_score/cot_monitor/BaseVerifier.py` | `compute_score` — base reward verifier |
| `verl/trainer/ppo/metric_utils.py` | `compute_data_metrics` — training metric logging |
| `verl/data_preprocessing/` | Dataset-specific preprocessing scripts |

### Configuration System

Config is managed by **Hydra** with YAML files in `verl/trainer/config/`. Key classmate-specific parameters in `qwen_classmate_cot_ppo_trainer.yaml`:

- `reward_model.classmate_model_name_or_path_list`: list of classmate model paths
- `reward_model.classmate_reward_type`: `vanilla_reward`, `remove_wo_cot`, `random_truncate`
- `reward_model.use_classmate_condition`: `always`, `no_classmate_when_main_incorrect`, `neg_classmate_when_main_incorrect`
- `reward_model.token_level_classmate_reward_mode`: `classmate_partial` or `all`
- `reward_model.rollout_is_threshold`: importance sampling threshold (default 4.0)
- `algorithm.adv_estimator`: `grpo`, `gdpo`, `grpo_main_classmate_separated`

### Supported Algorithms

- `grpo` — standard GRPO baseline
- `gdpo` — classmate training (main algorithm)
- `grpo_main_classmate_separated` — separate advantage for main and classmate tokens
- Also: `gae`, `drgrpo`, `rloo`, `remax` (inherited from upstream verl)

### Reward System

`SycophancyClassmateCoTRewardManager` orchestrates:
1. Base reward via `compute_score()` (format + correctness)
2. Classmate reward: iterate classmate samples → per-sample reward → average per classmate model → weighted average across classmate models
3. Optional Tinker LLM judge for sycophancy evaluation (`user_tinker_llm_judge=True`)

### Dataset Preprocessing

Dataset-specific scripts live in `verl/data_preprocessing/`. After modifying a dataset script, run `bash my_scripts/prepare_data.sh`. Datasets used: `helpful_instructions`, `anthropic_hh_rlhf`, `mmlu_sycophancy`, `gsm8k`, `hendrycks_math`.

### Debugging Notes

- Ray actors cannot use `pdb`; use print/logging instead
- Startup takes significant time due to model loading; plan accordingly
- FSDP checkpoints must be merged before uploading to HuggingFace
- Classmate model mapping JSON (`outputs/host_classmate_models/classmate_model_mapping.json`) must be updated after hosting classmate models
