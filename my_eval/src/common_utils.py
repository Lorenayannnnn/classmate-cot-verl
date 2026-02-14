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

def prepare_wandb(configs):
    # Check if parameter passed or if set within environ
    if configs.training_args.use_wandb and (len(configs.training_args.wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )) and configs.training_args.do_train:
        configs.training_args.use_wandb = True
        # Only overwrite environ if wandb param passed
        if len(configs.training_args.wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = configs.training_args.wandb_project
        if len(configs.training_args.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = configs.training_args.wandb_watch
        if len(configs.training_args.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = configs.training_args.wandb_log_model
        configs.wandb_run_name = configs.training_args.output_dir.split("/")[-1]
        wandb.init(project=os.environ["WANDB_PROJECT"], name=configs.wandb_run_name)
    else:
        configs.training_args.use_wandb = False
    return configs


def load_config_and_setup_output_dir(configs):
    short_dataset_name = configs.data_args.dataset_name.split("/")[-1]
    short_model_name = configs.model_args.model_name_or_path.split("/")[-1]

    outputs_root_dir = "outputs_eval"

    if configs.running_args.exp_type == "inference_main_model":
        dataset_subset_name = f"/{configs.data_args.dataset_subset_name}" if configs.data_args.dataset_subset_name is not None else ""
        configs.running_args.output_dir = f"{outputs_root_dir}/{configs.running_args.exp_type}/{short_model_name}/step_{configs.model_args.main_model_step_idx}/{short_dataset_name}{dataset_subset_name}"

        if "mmlu_sycophancy" in configs.data_args.dataset_name:
            configs.data_args.dataset_name = "data/mmlu_sycophancy_new/test.parquet"
        elif "helpful_instructions" in configs.data_args.dataset_name:
            configs.data_args.dataset_name = "data/helpful_instructions/test.parquet"
        elif "anthropic_hh_rlhf" in configs.data_args.dataset_name:
            configs.data_args.dataset_name = "data/anthropic_hh_rlhf/test.parquet"

    elif configs.running_args.exp_type == "cot_utility_to_classmate":
        short_main_model_name = configs.data_args.main_model_name_or_path.split("/")[-1]
        dataset_subset_name = f"/{configs.data_args.dataset_subset_name}" if configs.data_args.dataset_subset_name is not None else ""
        if not configs.data_args.wo_cot:
            configs.data_args.dataset_name = f"{outputs_root_dir}/inference_main_model/{short_main_model_name}/step_{configs.data_args.main_model_step_idx}/{short_dataset_name}{dataset_subset_name}/preds.jsonl"
            configs.running_args.output_dir = f"{outputs_root_dir}/{configs.running_args.exp_type}/{short_main_model_name}/step_{configs.data_args.main_model_step_idx}/{short_model_name}/{short_dataset_name}{dataset_subset_name}"
        else:
            configs.running_args.output_dir = f"{outputs_root_dir}/{configs.running_args.exp_type}/{short_main_model_name}/wo_cot/{short_model_name}/{short_dataset_name}{dataset_subset_name}"
    elif configs.running_args.exp_type == "reward_classmate_model":
        raise NotImplementedError
        # if args.main_model_step_idx is not None:
        #     configs.data_args.main_model_step_idx = args.main_model_step_idx
        # if args.model_name_or_path is not None:
        #     configs.model_args.model_name_or_path = args.model_name_or_path
        #     short_model_name = configs.model_args.model_name_or_path.split("/")[-1]
        # short_main_model_name = configs.data_args.main_model_name_or_path.split("/")[-1]
        # configs.data_args.dataset_name = f"{outputs_root_dir}/cot_utility_to_classmate/{short_main_model_name}/step_{configs.data_args.main_model_step_idx}/{short_dataset_name}/correct_preds.jsonl"
        # configs.running_args.output_dir = f"{outputs_root_dir}/cot_utility_to_classmate/{short_main_model_name}/step_{configs.data_args.main_model_step_idx}/{short_dataset_name}"
    else:
        raise ValueError(f"Unknown exp_type {configs.running_args.exp_type}")

    output_dir = configs.running_args.output_dir
    #
    output_config_fn = os.path.join(output_dir, "configs.yaml")
    prepare_folder(output_dir)
    OmegaConf.save(configs, output_config_fn)

    return configs


# def load_config_and_setup_output_dir(args):
#     base_configs = args.base_configs
#     overwrite_configs = args.overwrite_configs
#     if not os.path.exists(args.base_configs):
#         raise FileNotFoundError(f"Config file {args.base_config} does not exist")
#     if args.overwrite_configs is not None and not os.path.exists(args.overwrite_configs):
#         raise FileNotFoundError(f"Config file {args.overwrite_configs} does not exist")
#     configs = OmegaConf.load(base_configs)
#
#     output_dir = configs.training_args.output_dir
#
#     if overwrite_configs is not None:
#         overwrite_configs = OmegaConf.load(overwrite_configs)
#         output_dir = overwrite_configs.training_args.output_dir
#
#         # Merge base and overwrite configs
#         configs = OmegaConf.merge(configs, overwrite_configs)
#
#     if configs.training_args.resume_from_checkpoint is not None:
#         # Load configs from checkpoint
#         output_dir = os.path.dirname(configs.training_args.resume_from_checkpoint)
#         loaded_configs = OmegaConf.load(os.path.join(output_dir, "configs.yaml"))
#         loaded_configs.training_args.do_train = configs.training_args.do_train
#         loaded_configs.training_args.do_predict = configs.training_args.do_predict
#         loaded_configs.training_args.resume_from_checkpoint = configs.training_args.resume_from_checkpoint
#         configs = loaded_configs
#
#     # Prepare output directory
#     configs.training_args.output_dir = output_dir
#     prepare_folder(output_dir)
#     OmegaConf.save(configs, os.path.join(output_dir, "configs.yaml"))
#
#     # Save base and overwrite configs
#     # OmegaConf.save(configs, os.path.join(configs.training_args.output_dir, "configs", "base_configs.yaml"))
#     # OmegaConf.save(overwrite_configs, os.path.join(configs.training_args.output_dir, "configs", "overwrite_configs.yaml"))
#
#     return configs
