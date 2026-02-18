# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import random
import numpy as np
import ray
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.mismatch_helper import (
    compute_rollout_importance_weights
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async, compute_main_classmate_reward
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.reward_score.cot_monitor.monitor import process_main_cot_helper, monitor_cot_wrapper_w_tinker
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.my_utils import parse_out_main_cot_output
from verl.utils.reward_score.olmo_verifiers import process_code_output
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.classmate_workers import ClassmateWorker, ClassmateWorkerConfig


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    token_level_classmate_reward_mode: str = "all",
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns, group_reward_mean, group_reward_std, batch_main_rewards = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        data.batch["group_reward_mean"] = group_reward_mean
        data.batch["group_reward_std"] = group_reward_std
        data.batch["batch_main_rewards"] = batch_main_rewards
    elif adv_estimator == AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED:
        raise ValueError("Not supported; should use GDPO for multiple rewards")
        # Initialize the mask for GRPO calculation

        main_calculation_mask = data.batch["response_mask"]
        classmate_calculation_mask = data.batch["classmate_input_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_main_classmate_separated_outcome_advantage(
            main_token_level_rewards=data.batch["main_token_level_rewards"],
            classmate_token_level_rewards=data.batch["classmate_token_level_rewards"],
            # consistency_token_level_rewards=data.batch["consistency_token_level_rewards"],
            main_response_mask=main_calculation_mask,
            classmate_input_mask=classmate_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            token_level_classmate_reward_mode=token_level_classmate_reward_mode
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED_NON_NEG_CL:
        raise NotImplementedError("GRPO_MAIN_CLASSMATE_SEPARATED_NON_NEG_CL not implemented.")
        # Initialize the mask for GRPO calculation

        main_calculation_mask = data.batch["response_mask"]
        classmate_calculation_mask = data.batch["classmate_input_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_main_classmate_separated_non_neg_cl_outcome_advantage(
            main_token_level_rewards=data.batch["main_token_level_rewards"],
            classmate_token_level_rewards=data.batch["classmate_token_level_rewards"],
            main_response_mask=main_calculation_mask,
            classmate_input_mask=classmate_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            token_level_classmate_reward_mode=token_level_classmate_reward_mode
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GDPO:
        # Initialize the mask for GRPO calculation

        main_calculation_mask = data.batch["response_mask"]
        classmate_calculation_mask = data.batch["classmate_input_mask"]

        # Call compute_gdpo_outcome_advantage_separate with parameters matching its definition
        (
            main_advantages,
            classmate_advantages,
            main_group_reward_mean,
            main_group_reward_std,
            classmate_group_reward_mean,
            classmate_group_reward_std,
            batch_main_rewards,
            batch_classmate_rewards
        ) = core_algos.compute_gdpo_outcome_advantage_separate(
            main_token_level_rewards=data.batch["main_token_level_rewards"],
            classmate_token_level_rewards=data.batch["classmate_token_level_rewards"],
            # consistency_token_level_rewards=data.batch["consistency_token_level_rewards"],
            main_response_mask=main_calculation_mask,
            classmate_input_mask=classmate_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            token_level_classmate_reward_mode=token_level_classmate_reward_mode
        )
        data.batch["main_advantages"] = main_advantages
        data.batch["classmate_advantages"] = classmate_advantages
        # data.batch["advantages"] = main_advantages + classmate_advantages
        data.batch["returns"] = main_advantages + classmate_advantages      # Just for logging since we are not training critic model
        data.batch["main_group_reward_mean"] = main_group_reward_mean
        data.batch["main_group_reward_std"] = main_group_reward_std
        data.batch["weighted_classmate_group_reward_mean"] = classmate_group_reward_mean        # Note these are weighted
        data.batch["weighted_classmate_group_reward_std"] = classmate_group_reward_std  
        data.batch["batch_main_rewards"] = batch_main_rewards
        data.batch["batch_classmate_rewards"] = batch_classmate_rewards
        # data.batch["consistency_group_reward_mean"] = consistency_group_reward_mean
        # data.batch["consistency_group_reward_std"] = consistency_group_reward_std
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class ClassmateCoTRayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,

        # TODO modify
        classmate_cot_reward_configs=None,
        classmate_batch_size=None,
        classmate_free_cache_engine=None,
        classmate_use_vllm=None,
        classmate_num_return_sequences=None,
        classmate_generation_configs=None,
        classmate_vllm_configs=None,
        seed=None,
        enable_thinking=None,
        # host_classmate_name_to_url_json_fn=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.

            classmate_cot_reward_configs:
                - classmate_model_name_or_path_list: ["meta-llama/Llama-3.2-1B-Instruct"]
                - classmate_reward_weight: 1
                - classmate_reward_type: vanilla_reward
                - classmate_continue_mode: continue_cot
            classmate_generation_configs:
                - do_sample: False
                - max_new_tokens: 256
                - temperature: 0.7
                - top_p: 0.9
                - num_return_sequences: 1
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # Store classmate configurations
        self.classmate_cot_reward_configs = classmate_cot_reward_configs
        self.classmate_batch_size = classmate_batch_size
        self.classmate_free_cache_engine = classmate_free_cache_engine
        self.classmate_use_vllm = classmate_use_vllm
        self.classmate_generation_configs = classmate_generation_configs
        # self.host_classmate_name_to_url_json_fn = host_classmate_name_to_url_json_fn

        self.classmate_num_return_sequences = classmate_num_return_sequences
        OmegaConf.set_struct(self.classmate_generation_configs, False)
        if not self.classmate_generation_configs.do_sample:
            self.classmate_generation_configs.update({
                "temperature": 0.0,
                "top_k": -1,
                "top_p": 1.0,
                "seed": seed
            })

        if type(self.classmate_generation_configs["max_tokens"]) == str:
            self.classmate_generation_configs["max_tokens"] = int(eval(self.classmate_generation_configs["max_tokens"]))

        # self.classmate_tokenizers = {}

        if classmate_use_vllm:
            self.classmate_generation_configs.n = classmate_num_return_sequences
            self.classmate_generation_configs.seed = seed
            self.classmate_generation_configs.pop("do_sample")
        else:
            raise NotImplementedError("Only vLLM-based classmate is supported now.")
            # self.classmate_generation_configs.num_return_sequences = classmate_num_return_sequences
        OmegaConf.set_struct(self.classmate_generation_configs, True)

        self.classmate_vllm_configs = classmate_vllm_configs

        # Initialize seeded random number generator for reproducibility
        self.rng = random.Random(seed)

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # for CoT monitoring
        if self.config.reward_model.get("do_cot_monitor") is not None and self.config.reward_model.do_cot_monitor:
            # Initialize openai client
            from openai import OpenAI
            from keys import OPENAI_KEY
            self.openai_client = OpenAI(api_key=OPENAI_KEY)
            self._monitor_verifier = None
            self._monitor_verifier_key = None

        assert enable_thinking is not None, "Specify enable_thinking when initializing ClassmateCoTRayPPOTrainer."
        self.enable_thinking = enable_thinking

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _get_monitor_verifier(self, data_sources=None):
        # Optimize this later
        is_mmlu = False
        is_mmlu_pro = False
        data_source_values = []
        if data_sources is not None:
            if isinstance(data_sources, (list, tuple, np.ndarray)):
                data_source_values = data_sources
            else:
                data_source_values = [data_sources]

            try:
                is_mmlu = any("mmlu" in str(src) for src in data_source_values)
                is_mmlu_pro = any("mmlu_pro" in str(src) for src in data_source_values)
            except TypeError:
                is_mmlu = False
                is_mmlu_pro = False

        cache_key = (is_mmlu, is_mmlu_pro)
        if self._monitor_verifier is not None and self._monitor_verifier_key == cache_key:
            return self._monitor_verifier

        from verl.utils.reward_score.cot_monitor.BaseVerifier import get_monitor_verifier

        verifier = get_monitor_verifier(data_source=data_source_values)

        self._monitor_verifier = verifier
        self._monitor_verifier_key = cache_key
        return verifier

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        all_main_rewards = []
        all_monitor_scores = []
        all_monitor_explanations = []

        print("🐛🐛🐛 Start validating")
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            test_batch.non_tensor_batch["decoded_responses"] = np.array(self.tokenizer.batch_decode(
                test_batch.batch["responses"], skip_special_tokens=True
            ))

            # modify cot monitor
            if self.config.reward_model.get("do_cot_monitor") is not None and self.config.reward_model.do_cot_monitor:
                print("Performing CoT monitoring...")

                data_sources_batch = test_batch.non_tensor_batch.get("data_source")
                verifier = self._get_monitor_verifier(data_sources=data_sources_batch)
                if verifier is None:
                    raise ValueError("Monitor verifier could not be initialized from data_source.")

                # Extract CoTs, questions, and hints
                all_main_cots, all_raw_questions, all_hint_strs = [], [], []
                monitor_prompts = []
                for item, response in zip(test_batch, test_batch.non_tensor_batch["decoded_responses"]):
                    truncated_main_cot, _, _, _, _, _ = process_main_cot_helper(
                        self.tokenizer,
                        self.enable_thinking,
                        item.non_tensor_batch["data_source"],
                        response,
                        item.batch["responses"],  # token ids
                        None,  # response masks
                        self.classmate_cot_reward_configs.main_cot_keep_rate
                    )
                    # all_main_cots.append(parse_out_main_cot_output(response)[0])
                    all_main_cots.append(truncated_main_cot)
                    raw_question_for_monitor = item.non_tensor_batch["reward_model"]["raw_question_for_monitor"]
                    hint_str_for_monitor = item.non_tensor_batch["reward_model"].get("hint_str_for_monitor")
                    all_raw_questions.append(raw_question_for_monitor)
                    all_hint_strs.append(hint_str_for_monitor)

                    monitor_doc = {
                        "raw_question_for_monitor": raw_question_for_monitor,
                        "hint_str_for_monitor": hint_str_for_monitor,
                        "truncated_main_CoT": truncated_main_cot,
                    }
                    monitor_prompts.append(verifier.create_monitor_message(monitor_doc))

                # Execute monitor_cot calls in parallel
                # all_monitor_scores_batch, all_monitor_explanations_batch = monitor_cot_wrapper(
                #     monitor_prompts=monitor_prompts,
                #     monitor_reward_type=verifier.reward_type,
                #     openai_client=self.openai_client,
                #     model_name=self.config.reward_model["monitor_model_name"]
                # )
                all_monitor_scores_batch, all_monitor_explanations_batch = monitor_cot_wrapper_w_tinker(
                    monitor_prompts=monitor_prompts,
                )

                test_batch.non_tensor_batch["truncated_main_cot"] = np.array(all_main_cots)
                test_batch.non_tensor_batch["monitor_score"] = np.array(all_monitor_scores_batch)
                test_batch.non_tensor_batch["monitor_explanations"] = np.array(all_monitor_explanations_batch)
                test_batch.non_tensor_batch["monitor_reward_type"] = np.array([verifier.reward_type] * len(all_monitor_scores_batch))
                
                # Accumulate monitor results across all batches
                all_monitor_scores.extend(all_monitor_scores_batch)
                all_monitor_explanations.extend(all_monitor_explanations_batch)
                # print("CoT monitoring completed.")

            # Generate classmate continuations for validation
            # print("Generating classmate continuations for validation...")
            test_batch = self._generate_classmate_continuations(test_batch, is_eval=True)

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)

            # main_reward_tensor, classmate_reward_tensor, reward_extra_info
            # reward_tensor = result["reward_tensor"]
            # reward_tensor = result["main_reward_tensor"] + result["classmate_reward_tensor"] + result.get("consistency_reward_tensor", 0)
            reward_tensor = result["main_reward_tensor"] + result["classmate_reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

            all_main_rewards.extend(scores)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            # Determine core variable: prefer acc, then check for final_reward, fallback to reward
            # if "acc" in var2metric2val:
            #     core_var = "acc"
            # elif "final_reward" in var2metric2val:
            #     core_var = "final_reward"
            # else:
            #     core_var = "reward"
            
            # core_reward_vars = {"acc", "main_model_reward", "classmate_reward", "consistency_reward", "final_reward", "reward"}
            core_reward_vars = {"acc", "main_model_reward", "classmate_reward", "final_reward", "reward"}
            # TODO haha for code & test case generation
            code_test_case_reward_keys = [k for k in var2metric2val.keys() if "test_case_format_score" in k or "correct_code_score" in k]
            core_reward_vars.update(code_test_case_reward_keys)
            
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    # A metric goes to val-core if:
                    # 1. It's one of the core reward variables AND
                    # 2. It's a summary metric (mean/maj/best) at max sample size
                    if (
                        (var_name in core_reward_vars)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()
        
        if len(all_monitor_scores) > 0:
            # Convert to numpy arrays for computation
            monitor_scores = np.array(all_monitor_scores)
            main_model_reward = np.array(all_main_rewards)

            monitor_valid_mask = monitor_scores != 0
            main_reward_valid_mask = main_model_reward != 0
            valid_mask = monitor_valid_mask & main_reward_valid_mask

            monitor_scores = monitor_scores[valid_mask]
            main_model_reward = main_model_reward[valid_mask]

            monitor_scores = monitor_scores.astype(float)
            main_model_reward = main_model_reward.astype(float)

            verifier = self._get_monitor_verifier(data_sources=data_sources)
            if verifier is not None:
                monitor_metrics = verifier.compute_metrics(
                    predictions=monitor_scores.tolist(),
                    ground_truths=main_model_reward.tolist(),
                )
                metric_dict["val-core/monitor/total_monitor_valid_entries"] = np.sum(valid_mask)
                metric_dict["val-core/monitor/total_output_valid_entries"] = np.sum(main_reward_valid_mask)
                metric_dict["val-core/monitor/total_monitored_entries"] = len(all_monitor_scores)
                metric_dict.update({
                    f"val-core/monitor/{metric_name}": float(metric_val)
                    for metric_name, metric_val in monitor_metrics.items()
                })

        print("🐛🐛🐛 End validating")
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # TODO modify: create classmates, one worker per model
        model_paths = self.classmate_cot_reward_configs.classmate_model_name_or_path_list
        # Read in host_classmate_name_to_url_json_fn
        # with open(self.host_classmate_name_to_url_json_fn, "r") as f:
        #     host_classmate_name_to_url = json.load(f)
        # Deprecated. Now using TogetherAI
        for model_idx, model_path in enumerate(model_paths):
            # All classmate workers share the same resource pool
            # For different resource pools per model, modify ResourcePoolManager mapping
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ClassmateWorker)

            # if model_path not in host_classmate_name_to_url:
            #     raise ValueError(f"Model path {model_path} not found in host_classmate_name_to_url mapping.")

            # Create separate config for each model
            classmate_config = ClassmateWorkerConfig(
                model_name_or_path=model_path,
                model_index=model_idx,
                classmate_batch_size=self.classmate_batch_size,
                classmate_free_cache_engine=self.classmate_free_cache_engine,
                classmate_use_vllm=self.classmate_use_vllm,
                # max_new_tokens=self.classmate_generation_configs.max_new_tokens,
                max_tokens=self.classmate_generation_configs.max_tokens,
                generation_config=self.classmate_generation_configs,
                vllm_config=self.classmate_vllm_configs,
                enable_thinking=self.enable_thinking,

                main_model_max_tokens=self.config.data.max_response_length,         # for padding for classmate - main importance ratio
                # Deprecated. Now using TogetherAI
                # api_host=host_classmate_name_to_url[model_path]["local_host"],
                # api_port=host_classmate_name_to_url[model_path]["port"],
            )

            # Create unique role name for each classmate model worker group
            classmate_role_name = f"{str(Role.ClassmateWorker)}_{model_idx}"

            classmate_worker_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ClassmateWorker],
                config=classmate_config,
                role=classmate_role_name,
            )
            self.resource_pool_to_cls[resource_pool][classmate_role_name] = classmate_worker_cls

            # self.classmate_tokenizers[model_idx] = AutoTokenizer.from_pretrained(model_path.replace("-Turbo", ""))

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # TODO modify: Initialize classmate workers using VERL's worker group pattern
        # Now there should be enough GPU memory available since actor's vLLM is offloaded
        self.classmate_workers = []
        model_paths = self.classmate_cot_reward_configs.classmate_model_name_or_path_list
        for model_idx, model_path in enumerate(model_paths):
            classmate_role_name = f"{str(Role.ClassmateWorker)}_{model_idx}"
            classmate_wg = all_wg[classmate_role_name]
            classmate_wg.init_model()
            self.classmate_workers.append(classmate_wg)
        print(f"Initialized {len(self.classmate_workers)} classmate")

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    # def _find_token_subsequence_for_string(self, token_ids, target_string):
    #     """
    #     Find the shortest subsequence of token_ids that decodes to target_string.

    #     Args:
    #         token_ids: Sequence of token IDs (list or tensor)
    #         target_string: Target string to match

    #     Returns:
    #         int: Index i such that tokenizer.decode(token_ids[:i]) == target_string.
    #              Returns -1 if no match found.
    #     """
    #     right_ptr = 1
    #     while right_ptr < len(token_ids):
    #         decoded = self.tokenizer.decode(token_ids[:right_ptr], skip_special_tokens=True)
    #         # Exact match found - try to extend to include trailing whitespace tokens
    #         if decoded.strip() == target_string.strip():
    #             # Check if we can include additional whitespace tokens
    #             while right_ptr < len(token_ids):
    #                 decoded = self.tokenizer.decode(token_ids[:right_ptr + 1], skip_special_tokens=True)
    #                 # Stop if we exceed target length or lose the match
    #                 if len(decoded) > len(target_string) or decoded.strip() != target_string.strip():
    #                     break
    #                 right_ptr += 1
    #             return right_ptr

    #         # Early exit: no early exit because there might be a lot of whitespace characters prepended
    #         # if len(decoded) > len(target_string):
    #             # Decoded is already longer than target, won't find match
    #             # break

    #         right_ptr += 1

    #     return -1

    # def _old_process_main_cot_helper(self, data_source, main_cot, main_pred_token_ids, main_response_mask, keep_ratio):
    #     """
    #     Split CoT from main model by all whitespace characters (e.g., whitespace, tab, newline), and keep a portion of tokens
    #     """
    #     # TODO haha double check for code_contests_modify_code
    #     if data_source == "code_contests_modify_code":
    #         breakpoint()
    #         reason_section_name = "### Reasoning"
    #         test_case_section_name = "### Test Inputs and Outputs"
    #         main_cot = reason_section_name + "\n" + main_cot.split(reason_section_name)[-1].split(test_case_section_name)[0].strip()
    #     # elif data_source == "code_stdio" or data_source == "code":
    #     #     used in olmo3 CodeVerifier
    #         # main_pred = process_code_output(main_pred)
    #
    #     if self.enable_thinking:
    #         main_cot, _, _ = self.parse_out_main_cot_output(main_cot)
    #
    #     _SPLIT_WS = re.compile(r"\S+|\s+")
    #     num_to_keep = int(len(main_cot.split()) * keep_ratio)  # counts non-whitespace tokens
    #     if num_to_keep <= 0:
    #         # Keep the entire main CoT when it's too short
    #         return main_cot, main_response_mask.tolist()
    #
    #     kept = []
    #     count = 0
    #     for m in _SPLIT_WS.finditer(main_cot):
    #         t = m.group(0)
    #         # whitespace tokens start with a whitespace char
    #         if t[0].isspace():
    #             kept.append(t)
    #         else:
    #             count += 1
    #             if count > num_to_keep:
    #                 break
    #             kept.append(t)
    #
    #     truncated_text = "".join(kept)
    #
    #     if self.enable_thinking:
    #         truncated_text = "<think>" + truncated_text
    #
    #     # Find the token index that corresponds to the truncated text
    #     token_index = self._find_token_subsequence_for_string(main_pred_token_ids, truncated_text)
    #
    #     if token_index == -1:
    #         # # store to output file
    #         # with open("debug_truncation_failure.txt", "a") as f:
    #         #     f.write(f"Data source: {data_source}\n")
    #         #     f.write(f"Original main_pred: {main_pred}\n")
    #         #     f.write(f"Truncated text: {truncated_text}\n")
    #         #     f.write(f"Token IDs: {main_pred_token_ids.tolist()}\n")
    #         #     f.write("\n" + "="*80 + "\n\n")
    #         raise ValueError(
    #             f"Could not find token subsequence matching truncated text.\n"
    #             f"Truncated text: {truncated_text}\n"
    #             f"Token IDs: {main_pred_token_ids.tolist()}"
    #         )
    #
    #     # print("🐛🐛🐛 original main_pred", main_pred)
    #     # print("🐛🐛🐛 truncated_text", truncated_text)
    #     # print("🐛🐛🐛 token_index", token_index)
    #     # print("🐛🐛🐛 main_response_mask", main_response_mask)
    #     # print("🐛🐛🐛 classmate_input_mask", main_response_mask[:token_index].tolist())
    #     # print("🐛🐛🐛 final_classmate_input_mask", main_response_mask[:token_index].tolist() + [0] * (len(main_response_mask) - token_index))
    #
    #     return truncated_text, main_response_mask[:token_index].tolist() + [0] * (len(main_response_mask) - token_index)

    # def derangement(self, lst):
    #     n = len(lst)
    #     if n <= 1:
    #         raise ValueError("Derangement not possible")
    #
    #     perm = lst[:]
    #
    #     for i in range(n - 1):
    #         j = random.randrange(i + 1, n)
    #         perm[i], perm[j] = perm[j], perm[i]
    #
    #     # Fix last element if needed
    #     if perm[-1] == lst[-1]:
    #         perm[-1], perm[-2] = perm[-2], perm[-1]
    #
    #     return perm

    def derange_two_lists(self, lst1, lst2):
        if len(lst1) != len(lst2):
            raise ValueError("Lists must have the same length")

        n = len(lst1)
        if n <= 1:
            raise ValueError("Derangement not possible")

        idx = list(range(n))

        # Generate derangement of indices
        for i in range(n - 1):
            j = self.rng.randrange(i + 1, n)
            idx[i], idx[j] = idx[j], idx[i]

        if idx[-1] == n - 1:
            idx[-1], idx[-2] = idx[-2], idx[-1]

        # Apply permutation
        new1 = [lst1[i] for i in idx]
        new2 = [lst2[i] for i in idx]

        return new1, new2

    def _prepare_classmate_batch(self, data: DataProto, is_eval: bool):
        """
        Prepare batch data for classmate generation. Each worker will handle its own tokenization.
        This part is only handling main model's CoT. Prompt/question before the CoT will be handled separately when sampling from classmates
        given we need to apply chat template
        """
        if "response_mask" not in data.batch.keys() and is_eval:
            data.batch["response_mask"] = compute_response_mask(data)

        all_actor_responses = data.non_tensor_batch["decoded_responses"]
        max_response_len = max(len(data.batch["response_mask"][i]) for i in range(len(all_actor_responses)))

        all_prompts = data.non_tensor_batch["raw_prompt"]
        all_processed_actor_responses = []
        all_processed_actor_responses_token_ids = []
        all_classmate_input_mask = []
        all_keep_rates = []
        all_start_with_thinks = []
        all_end_with_thinks = []

        all_other_prompts = None
        all_other_prompt_gts = None

        # if "help_other" in self.classmate_cot_reward_configs.classmate_reward_type:
        #     all_processed_actor_responses = all_actor_responses
        #     all_classmate_input_mask = [[1] * max_response_len] * len(all_prompts)
        #     all_keep_rates = [1.0] * len(all_prompts)
        #     all_reward_model_info = data.non_tensor_batch["reward_model"].tolist()
        #     if self.classmate_cot_reward_configs.classmate_reward_type == "help_other_questions":       # a random, different in-batch question
        #         all_other_prompt_gts = [info["ground_truth"] for info in all_reward_model_info]
        #         all_other_prompts, all_other_prompt_gts = self.derange_two_lists(all_prompts, all_other_prompt_gts)
        #     elif self.classmate_cot_reward_configs.classmate_reward_type == "help_other_similar_questions":     # preprocessed, paired question from the same category/type
        #         all_other_prompts = []
        #         all_other_prompt_gts = []
        #         for info in all_reward_model_info:
        #             all_other_prompts.append(info["paired_similar_q"])
        #             all_other_prompt_gts.append(info["paired_similar_q_gt"])
        if self.classmate_cot_reward_configs.classmate_reward_type in ["vanilla_reward", "random_truncate", "remove_wo_cot"]:
            # Note: assuming the prompt has sth like Let\'s think step by step and output the final answer after "####".
            # all_prompts = self.tokenizer.batch_decode(
            #     data.non_tensor_batch["raw_prompt"], skip_special_tokens=True
            # )
            # all_prompts = [message[0]["content"] for message in data.non_tensor_batch["raw_prompt"]]

            # print("🐛🐛🐛 raw_prompt cnt", len(data.non_tensor_batch["raw_prompt"]))
            # print("🐛🐛🐛 raw_messages", data.non_tensor_batch["raw_prompt"])
            # print("🐛🐛🐛 raw_prompts", all_prompts)

            # print("🐛 sample prompt", all_prompts[0])
            # print("🐛 sample prompt id", data.batch["prompts"][0])
            # print("🐛 sample response", all_actor_responses[0])
            # print("🐛 tokenizer padding token", self.tokenizer.pad_token)

            assert self.classmate_cot_reward_configs.classmate_continue_mode == "continue_cot", "Currently only support continue_cot mode."

            data_source_list = data.non_tensor_batch.get("data_source")

            for res_idx, response in enumerate(all_actor_responses):
                if is_eval or self.classmate_cot_reward_configs.classmate_reward_type == "vanilla_reward" or self.classmate_cot_reward_configs.classmate_reward_type == "remove_wo_cot":
                    # Use fixed dropout rate during evaluation or for vanilla reward
                    keep_rate = self.classmate_cot_reward_configs.main_cot_keep_rate
                    # if "code" in data_source_list[res_idx]:
                    #     print(f"🐛🐛🐛 prompt {all_prompts[res_idx]}")
                    #     print(f"🐛🐛🐛 original response {processed_response}")
                        # print(f"🐛🐛🐛 sample modified_response {modified_response}")
                elif self.classmate_cot_reward_configs.classmate_reward_type == "random_truncate":
                    keep_rate = self.rng.uniform(self.classmate_cot_reward_configs.main_cot_keep_rate_min, self.classmate_cot_reward_configs.main_cot_keep_rate_max)
                else:
                    raise NotImplementedError(f"{self.classmate_cot_reward_configs.classmate_reward_type} not implemented yet.")

                truncated_main_cot, truncated_pred_token_ids, _, classmate_input_mask, end_with_think, start_with_think = process_main_cot_helper(
                    self.tokenizer,
                    self.enable_thinking,
                    data_source_list[res_idx],
                    response,
                    data.batch["responses"][res_idx],       # token ids
                    data.batch["response_mask"][res_idx],   # response masks
                    keep_rate
                )
                all_processed_actor_responses.append(truncated_main_cot)
                all_keep_rates.append(keep_rate)
                all_end_with_thinks.append(end_with_think)
                all_start_with_thinks.append(start_with_think)

                # Pad mask to max_response_len to ensure all masks have the same length
                padded_mask = classmate_input_mask + [0] * (max_response_len - len(classmate_input_mask))
                all_classmate_input_mask.append(padded_mask)
                padded_truncated_pred_token_ids = truncated_pred_token_ids + [self.tokenizer.pad_token_id] * (max_response_len - len(truncated_pred_token_ids))
                # print("🐛🐛🐛 truncated_pred_token_ids", len(truncated_pred_token_ids))
                # print("🐛🐛🐛 padded_truncated_pred_token_ids", len(padded_truncated_pred_token_ids))
                all_processed_actor_responses_token_ids.append(padded_truncated_pred_token_ids)


        classmate_non_tensor_batch = {
            "raw_prompts": np.array(all_prompts),
            "main_model_responses": np.array(all_processed_actor_responses),
            "main_model_response_token_ids": np.array(all_processed_actor_responses_token_ids),
            "keep_rates": np.array(all_keep_rates),
            "end_with_thinks": np.array(all_end_with_thinks),
            "start_with_thinks": np.array(all_start_with_thinks),
            "classmate_input_mask": np.array(all_classmate_input_mask),
            # "prompt_plus_actor_responses": np.array(all_prompt_plus_actor_responses)
        }

        # with open("debug.txt", "a") as f:
        #     f.write(f"all_prompts: {all_prompts}\n")
        #     f.write(f"all_processed_actor_responses: {all_processed_actor_responses}\n")
        #     f.write(f"all_keep_rates: {all_keep_rates}\n")
        #     f.write(f"all_end_with_thinks: {all_end_with_thinks}\n")
        #     f.write(f"all_start_with_thinks: {all_start_with_thinks}\n")
        #     f.write(f"all_classmate_input_mask: {all_classmate_input_mask}\n")
        #     f.write("\n" + "="*80 + "\n\n")

        if all_other_prompts is not None:
            classmate_non_tensor_batch["all_other_prompts"] = np.array(all_other_prompts)
            classmate_non_tensor_batch["all_other_prompt_gts"] = np.array(all_other_prompt_gts)

        # print("🐛 Sample classmate input from prepare batch", all_processed_actor_responses[0])
        # print("🐛 Sample classmate input from prepare batch", all_processed_actor_responses[1])
        # print("🐛 all_classmate_input_mask", all_classmate_input_mask)
        # print("🐛 all_classmate_input_mask", all_classmate_input_mask[1])

        # All masks are now padded to max_response_len, so they can be safely converted to a tensor
        return DataProto(batch=None, non_tensor_batch=classmate_non_tensor_batch)

    def _prepare_help_other_qs_cot_classmate_batch(self, data: DataProto, is_eval: bool):
        """
        Prepare batch data for classmate generation in help_other_questions mode.
        Each worker will handle its own tokenization.
        """
        def derangement(lst):
            n = len(lst)
            if n <= 1:
                raise ValueError("Derangement not possible")

            perm = lst[:]

            for i in range(n - 1):
                j = random.randrange(i + 1, n)
                perm[i], perm[j] = perm[j], perm[i]

            # Fix last element if needed
            if perm[-1] == lst[-1]:
                perm[-1], perm[-2] = perm[-2], perm[-1]

            return perm

        all_prompts = data.non_tensor_batch["raw_prompt"]
        all_actor_responses = self.tokenizer.batch_decode(
            data.batch["responses"], skip_special_tokens=True
        )
        max_response_len = max(len(data.batch["response_mask"][i]) for i in range(len(all_actor_responses)))
        all_other_prompts = derangement(all_prompts)
        all_classmate_input_mask = [1] * len(max_response_len)
        all_keep_rates = [1.0] * len(all_prompts)

        classmate_non_tensor_batch = {
            "raw_prompts": np.array(all_other_prompts),
            "main_model_responses": np.array(all_actor_responses),
            "keep_rates": np.array(all_keep_rates),
        }

        return torch.tensor(all_classmate_input_mask), DataProto(batch=None, non_tensor_batch=classmate_non_tensor_batch)



    def _generate_classmate_continuations(self, batch: DataProto, is_eval=False) -> DataProto:
        """
        Launch sampling on each classmate model in parallel and collect results.

        Expects each worker group to return a result with `.non_tensor_batch["classmate_outputs"]`
        where `classmate_output` has shape (bsz_per_worker, num_return_sequences).

        For worker groups with multiple workers (data parallel), results are collected
        and concatenated across workers.

        Reshape (num_classmates, bsz, num_return_sequences) -> (bsz, num_classmates, num_return_sequences)
        """
        # Build the shared input for all classmates
        classmate_batch = self._prepare_classmate_batch(batch, is_eval)

        keep_rates = classmate_batch.non_tensor_batch.pop("keep_rates")
        all_other_prompt_gts = classmate_batch.non_tensor_batch.pop("all_other_prompt_gts", None)

        bsz = len(batch)

        num_models = len(self.classmate_workers)

        # Set is_eval in meta_info for dispatch compatibility
        classmate_batch.meta_info["is_eval"] = is_eval

        futures = []  # list[(classmate_idx, future, pad_size)]
        for classmate_idx, worker_group in enumerate(self.classmate_workers):
            size_divisor = getattr(worker_group, "world_size", 1)
            if size_divisor > 1:
                # Pad the entire batch once to be divisible by world_size
                # Worker group internally handles distribution across workers
                classmate_batch_padded, pad_size = pad_dataproto_to_divisor(classmate_batch, size_divisor, pad_val=None)
                classmate_batch_padded.meta_info["is_eval"] = is_eval
                fut = worker_group.generate_classmate_continuations(data=classmate_batch_padded)
                futures.append((classmate_idx, fut, pad_size))
            else:
                # No padding needed for single worker
                fut = worker_group.generate_classmate_continuations(data=classmate_batch)
                futures.append((classmate_idx, fut, 0))

        # Resolve all models in parallel (DataProtoFuture -> DataProto)
        resolved = [f.get() for _, f, _ in futures]

        # Collect results - use NumPy array for efficient assignment
        classmate_prompts = np.empty((bsz, num_models), dtype=object)
        outputs = np.empty((bsz, num_models, self.classmate_num_return_sequences), dtype=object)
        classmate_response_length = np.empty((bsz, num_models, self.classmate_num_return_sequences), dtype=np.int32)
        classmate_prompt_logprobs_list = np.empty((bsz, num_models, self.classmate_num_return_sequences, self.config.data.max_response_length))     # pad to max_response_length for easier main - classmate tensor conversion later
        classmate_prompt_logprobs_mask_list = np.empty((bsz, num_models, self.classmate_num_return_sequences, self.config.data.max_response_length), dtype=bool)
        for (classmate_idx, _, pad_size), result_dp in zip(futures, resolved):
            cls_prompts = result_dp.non_tensor_batch["classmate_prompts"]  # (bsz_padded, )
            cls_out = result_dp.non_tensor_batch["classmate_outputs"]  # (bsz_padded, num_return_sequences)
            cls_out_lens = result_dp.non_tensor_batch["classmate_output_lens"]  # (bsz_padded, num_return_sequences)

            cls_prompt_logprobs = result_dp.non_tensor_batch["classmate_prompt_logprobs"]  # (bsz_padded, num_return_sequences, seq_len)
            cls_prompt_logprobs_mask = result_dp.non_tensor_batch["classmate_prompt_logprobs_mask"]  # (bsz_padded, num_return_sequences, seq_len)

            assert len(cls_out) == bsz, (
                f"Expected {bsz} outputs from model {classmate_idx}, got {len(cls_out)}"
            )
            outputs[:, classmate_idx] = cls_out
            classmate_prompts[:, classmate_idx] = cls_prompts
            classmate_response_length[:, classmate_idx] = cls_out_lens
            classmate_prompt_logprobs_list[:, classmate_idx] = cls_prompt_logprobs
            classmate_prompt_logprobs_mask_list[:, classmate_idx] = cls_prompt_logprobs_mask

        # Expected shape: (bsz, num_models, num_return_sequences)
        assert outputs.shape == (bsz, num_models, self.classmate_num_return_sequences), \
            f"Expected shape {(bsz, num_models, self.classmate_num_return_sequences)}, got {batch.non_tensor_batch['classmate_outputs'].shape}"
        batch.non_tensor_batch["classmate_prompts"] = classmate_prompts
        batch.non_tensor_batch["classmate_outputs"] = outputs
        
        batch.batch["classmate_prompt_logprobs"] = torch.tensor(classmate_prompt_logprobs_list)
        batch.batch["classmate_prompt_logprobs_mask"] = torch.tensor(classmate_prompt_logprobs_mask_list)

        batch.non_tensor_batch["classmate_response_length"] = np.array(classmate_response_length)
        batch.non_tensor_batch["classmate_max_tokens_len"] = np.array([self.classmate_generation_configs.max_tokens] * bsz)
        batch.non_tensor_batch["main_keep_rates"] = keep_rates
        if all_other_prompt_gts is not None:
            batch.non_tensor_batch["all_other_prompt_gts"] = all_other_prompt_gts

        batch.batch["classmate_input_mask"] = torch.tensor(classmate_batch.non_tensor_batch["classmate_input_mask"], dtype=torch.bool)

        return batch

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        """Compute rollout importance sampling weights and mismatch metrics, conditionally add weights to batch.

        This method computes IS weights to correct for distribution mismatch between
        rollout policy and training policy. It always computes metrics when enabled, but
        only adds weights to batch if algorithm.rollout_is is True.

        Args:
            batch: DataProto containing old_log_probs, rollout_log_probs, response_mask

        Returns:
            Tuple of (updated_batch, metrics) where:
                - updated_batch: Batch with rollout_is_weights added (if rollout_is=True)
                - metrics: Dictionary of IS and mismatch metrics (all with mismatch/ prefix)
        """
        # Compute rollout IS weights if enabled and data is available
        # rollout_is_threshold is the main on/off switch
        assert self.config.algorithm.rollout_is_threshold is not None, "rollout_is_threshold must be set to compute IS weights"
        assert "rollout_log_probs" in batch.batch, "rollout_log_probs must be in batch to compute IS weights"

        assert "classmate_prompt_logprobs" in batch.batch, "classmate_prompt_logprobs must be in batch to compute classmate IS weights"
        assert "classmate_prompt_logprobs_mask" in batch.batch, "classmate_prompt_logprobs_mask must be in batch to compute classmate IS weights"

        # if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
        rollout_is_weights, rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=batch.batch["old_log_probs"],
            rollout_log_prob=batch.batch["rollout_log_probs"],
            response_mask=batch.batch["response_mask"],
            rollout_is_level=self.config.algorithm.rollout_is_level,
            rollout_is_mode=self.config.algorithm.rollout_is_mode,
            rollout_is_threshold=self.config.algorithm.rollout_is_threshold,
            rollout_is_threshold_lower=self.config.algorithm.rollout_is_threshold_lower,
            rollout_is_veto_threshold=self.config.algorithm.rollout_is_veto_threshold,
        )

        # classmate_rollout_is_weights = None
        # classmate_rollout_is_metrics = {}
        # Assuming main and classmate has the same tokenizer

        assert batch.batch["classmate_prompt_logprobs"].shape[1] == 1 and batch.batch["classmate_prompt_logprobs"].shape[2] == 1, "Assuming has one classmate with each cl generating one seq"
        assert batch.batch["classmate_prompt_logprobs_mask"].shape[1] == 1 and batch.batch["classmate_prompt_logprobs_mask"].shape[2] == 1, "Assuming has one classmate with each cl generating one seq"

        batch.batch["classmate_prompt_logprobs"] = batch.batch["classmate_prompt_logprobs"].squeeze(2).squeeze(1)  # (bsz, seq_len)
        batch.batch["classmate_prompt_logprobs_mask"] = batch.batch["classmate_prompt_logprobs_mask"].squeeze(2).squeeze(1)  # (bsz, seq_len)

        classmate_rollout_is_weights, classmate_rollout_is_metrics = compute_rollout_importance_weights(
            old_log_prob=batch.batch["old_log_probs"],
            rollout_log_prob=batch.batch["classmate_prompt_logprobs"],
            response_mask=batch.batch["classmate_prompt_logprobs_mask"],
            rollout_is_level=self.classmate_cot_reward_configs.cl_rollout_is_level,
            rollout_is_mode=self.classmate_cot_reward_configs.cl_rollout_is_mode,
            rollout_is_threshold=self.classmate_cot_reward_configs.cl_rollout_is_threshold,
            rollout_is_threshold_lower=self.classmate_cot_reward_configs.cl_rollout_is_threshold_lower,
            rollout_is_veto_threshold=self.classmate_cot_reward_configs.cl_rollout_is_veto_threshold,
            rollout_is_weights_key_name="classmate_rollout_is_weights"
        )

        # Control: Should we apply weights to policy loss?
        # True = add weights to batch (actor will apply them)
        # False = don't add weights (metrics only, no loss modification)
        apply_weights = self.config.algorithm.get("rollout_is", False)

        if apply_weights:
            # Add IS weights to batch for distribution to workers
            batch = batch.union(rollout_is_weights)
            # Store classmate rollout IS weights separately for custom usage
            if classmate_rollout_is_weights is not None:
                batch = batch.union(classmate_rollout_is_weights)

        # Preserve existing classmate_mismatch/* keys for backward compatibility
        rollout_is_metrics.update(
            {f"classmate_{k}": v for k, v in classmate_rollout_is_metrics.items()}
        )
        # Add normalized mismatch/classmate_* keys for clearer logging
        classmate_metrics_normalized = {}
        for k, v in classmate_rollout_is_metrics.items():
            if k.startswith("mismatch/"):
                classmate_key = f"mismatch/classmate_{k[len('mismatch/'):] }"
            else:
                classmate_key = f"classmate_{k}"
            classmate_metrics_normalized[classmate_key] = v
        rollout_is_metrics.update(classmate_metrics_normalized)
        return batch, rollout_is_metrics

        # Return unchanged batch and empty metrics if IS is disabled
        # return batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.my_tracking import MyTracking

        logger = MyTracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            output_dir=self.config.trainer.default_local_dir,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.non_tensor_batch["decoded_responses"] = np.array(self.tokenizer.batch_decode(
                        batch.batch["responses"], skip_special_tokens=True
                    ))

                    # TODO: not doing monitor during training for now to save money
                    # if self.config.reward_model.get("do_cot_monitor") is not None and self.config.reward_model.do_cot_monitor:
                    #     from concurrent.futures import ThreadPoolExecutor, as_completed
                    #
                    #     # Prepare all monitor tasks
                    #     monitor_tasks = []
                    #     for item in batch:
                    #         # (skipping for now) 1. get enable_thinking from apply_chat_template_kwargs.enable_thinking (enable_thinking attribute may not exist)
                    #         # get main cot in-between think tag
                    #         main_cot_for_monitor = self.parse_out_main_cot(item.non_tensor_batch["decoded_responses"])
                    #
                    #         # call monitor: monitor_cot(user_message, hint_message, main_pred, monitor_template_name, openai_client, model_name="gpt-4o-mini"):
                    #         user_message = item.non_tensor_batch.get("reward_model", {}).get("raw_question_for_monitor", "")
                    #         hint_message = item.non_tensor_batch.get("reward_model", {}).get("hint_str_for_monitor", "")
                    #         monitor_tasks.append((user_message, hint_message, main_cot_for_monitor))
                    #
                    #     # Execute monitor_cot calls in parallel
                    #     all_monitor_use_hint = [None] * len(monitor_tasks)
                    #     all_monitor_explanations = [None] * len(monitor_tasks)
                    #
                    #     with ThreadPoolExecutor(max_workers=min(32, len(monitor_tasks))) as executor:
                    #         # Submit all tasks
                    #         future_to_idx = {
                    #             executor.submit(
                    #                 monitor_cot,
                    #                 user_msg,
                    #                 hint_msg,
                    #                 main_cot,
                    #                 self.config.reward_model.get("monitor_template_name"),
                    #                 self.openai_client,
                    #                 model_name=self.config.reward_model.get("monitor_model_name")
                    #             ): idx
                    #             for idx, (user_msg, hint_msg, main_cot) in enumerate(monitor_tasks)
                    #         }
                    #
                    #         # Collect results as they complete
                    #         for future in as_completed(future_to_idx):
                    #             idx = future_to_idx[future]
                    #             monitor_result_dict = future.result()
                    #             all_monitor_use_hint[idx] = monitor_result_dict['use_hint']
                    #             all_monitor_explanations[idx] = monitor_result_dict['explanation']
                    #
                    #     batch.non_tensor_batch["monitor_use_hint"] = np.array(all_monitor_use_hint)
                    #     batch.non_tensor_batch["monitor_explanations"] = np.array(all_monitor_explanations)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    with marked_timer("classmate_cot", timing_raw, color="magenta"):
                        batch = self._generate_classmate_continuations(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            # reward_tensor = self.rm_wg.compute_rm_score(batch)
                            # batch = batch.union(reward_tensor)
                            # main_reward_tensor, classmate_reward_tensor, consistency_reward_tensor = self.rm_wg.compute_rm_score(batch)
                            main_reward_tensor, classmate_reward_tensor, _ = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(main_reward_tensor)
                            batch = batch.union(classmate_reward_tensor)
                            # batch = batch.union(consistency_reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            # reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            # main_reward_tensor, classmate_reward_tensor, consistency_reward_tensor, reward_extra_infos_dict = compute_main_classmate_reward(batch, self.reward_fn)
                            main_reward_tensor, classmate_reward_tensor, reward_extra_infos_dict = compute_main_classmate_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            # reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            # main_reward_tensor, classmate_reward_tensor, consistency_reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            main_reward_tensor, classmate_reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        # batch.batch["token_level_scores"] = reward_tensor

                        if self.config.algorithm.adv_estimator in [AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED, AdvantageEstimator.GDPO, AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED_NON_NEG_CL]:
                            batch.batch["main_token_level_scores"] = main_reward_tensor
                            batch.batch["classmate_token_level_scores"] = classmate_reward_tensor
                            # batch.batch["consistency_token_level_scores"] = consistency_reward_tensor if consistency_reward_tensor is not None else 0
                        else:
                            # reward_tensor = main_reward_tensor + classmate_reward_tensor + consistency_reward_tensor if consistency_reward_tensor is not None else 0
                            reward_tensor = main_reward_tensor + classmate_reward_tensor
                            batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            if self.config.algorithm.adv_estimator in [AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED, AdvantageEstimator.GDPO, AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED_NON_NEG_CL]:
                                raise NotImplementedError("GRPO_MAIN_CLASSMATE_SEPARATED with KL in reward not implemented yet.")
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            if self.config.algorithm.adv_estimator in [AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED, AdvantageEstimator.GDPO, AdvantageEstimator.GRPO_MAIN_CLASSMATE_SEPARATED_NON_NEG_CL]:
                                batch.batch["main_token_level_rewards"] = batch.batch["main_token_level_scores"]
                                batch.batch["classmate_token_level_rewards"] = batch.batch["classmate_token_level_scores"]
                                # batch.batch["consistency_token_level_rewards"] = batch.batch["consistency_token_level_scores"]
                            else:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            token_level_classmate_reward_mode=self.classmate_cot_reward_configs.token_level_classmate_reward_mode,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
