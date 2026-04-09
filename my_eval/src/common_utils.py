import os
from random import random

import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from transformers import set_seed


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def prepare_folder(output_dir):
    """Prepare a folder for a file"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# def prepare_wandb(configs):
#     # Check if parameter passed or if set within environ
#     if configs.training_args.use_wandb and (len(configs.training_args.wandb_project) > 0 or (
#             "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
#     )) and configs.training_args.do_train:
#         configs.training_args.use_wandb = True
#         # Only overwrite environ if wandb param passed
#         if len(configs.training_args.wandb_project) > 0:
#             os.environ["WANDB_PROJECT"] = configs.training_args.wandb_project
#         if len(configs.training_args.wandb_watch) > 0:
#             os.environ["WANDB_WATCH"] = configs.training_args.wandb_watch
#         if len(configs.training_args.wandb_log_model) > 0:
#             os.environ["WANDB_LOG_MODEL"] = configs.training_args.wandb_log_model
#         configs.wandb_run_name = configs.training_args.output_dir.split("/")[-1]
#         wandb.init(project=os.environ["WANDB_PROJECT"], name=configs.wandb_run_name)
#     else:
#         configs.training_args.use_wandb = False
#     return configs


def load_config_and_setup_output_dir(configs):
    short_model_name = configs.model_args.model_name_or_path.split("/")[-1]

    outputs_root_dir = "outputs_eval"

    dataset_split_name = configs.data_args.dataset_split_name

    if "anthropic_sycophancy" in configs.data_args.dataset_name:
        raise NotImplementedError("anthropic_sycophancy not implemented")
        # configs.data_args.dataset_name = f"data/anthropic_sycophancy/{dataset_split_name}.parquet"
    elif "longer_response" in configs.data_args.dataset_name:
        configs.data_args.dataset_name = f"data/longer_response/{dataset_split_name}.parquet"
    elif "confidence" in configs.data_args.dataset_name:
        configs.data_args.dataset_name = f"data/confidence/{dataset_split_name}.parquet"
    elif "sycophancy" in configs.data_args.dataset_name:
        configs.data_args.dataset_name = f"data/sycophancy/{dataset_split_name}.parquet"
    elif "unsafe_compliance" in configs.data_args.dataset_name:
        configs.data_args.dataset_name = f"data/unsafe_compliance/{dataset_split_name}.parquet"
    elif "general_reward" in configs.data_args.dataset_name:
        configs.data_args.dataset_name = f"data/general_reward/{dataset_split_name}.parquet"
    else:
        raise ValueError(f"Unknown dataset {configs.data_args.dataset_name}")
    # elif "mmlu_sycophancy" in configs.data_args.dataset_name:
    #     configs.data_args.dataset_name = "data/mmlu_sycophancy_new/test.parquet"
    # elif "anthropic_hh_rlhf" in configs.data_args.dataset_name:
    #     configs.data_args.dataset_name = "data/anthropic_hh_rlhf/test.parquet"

    # Parse model repo name: {task}-{base_model}-{method}-{seed}
    # e.g. confidence-Qwen3-0.6B-baseline_all_tokens-seed_0
    task_part, remainder = short_model_name.split("-", 1)
    rest, seed_str = remainder.rsplit("-", 1)        # seed_str = "seed_0"
    base_model_str, method_str = rest.rsplit("-", 1)  # base_model = "Qwen3-0.6B", method = "baseline_all_tokens"

    dataset_subset_name = f"/{configs.data_args.dataset_subset_name}" if configs.data_args.dataset_subset_name is not None else ""
    step_idx = configs.model_args.main_model_step_idx
    if str(step_idx) == "base":
        # Base checkpoint is seed-independent — no seed subdirectory
        configs.running_args.output_dir = (
            f"{outputs_root_dir}"
            f"/{task_part}/{base_model_str}/{method_str}"
            f"/step_base/{dataset_split_name}{dataset_subset_name}"
        )
    else:
        configs.running_args.output_dir = (
            f"{outputs_root_dir}"
            f"/{task_part}/{base_model_str}/{method_str}"
            f"/step_{step_idx}/{seed_str}/{dataset_split_name}{dataset_subset_name}"
        )

    output_dir = configs.running_args.output_dir
    #
    output_config_fn = os.path.join(output_dir, "configs.yaml")
    prepare_folder(output_dir)
    OmegaConf.save(configs, output_config_fn)

    return configs
