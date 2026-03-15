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

# ── [MIND-FACE] ───────────────────────────────────────────────────────────────
# Copied from classmate_cot_ray_trainer.py and adapted for the mind-face baseline
# (Appendix A of https://arxiv.org/pdf/2511.11584).
#
# Key differences from the classmate trainer:
#   1. ORDER REVERSED: mind generates CoT FIRST, face generates answer SECOND.
#      (classmate trainer: face generates first, then classmate continues face's CoT)
#   2. Face model receives [prompt + mind_CoT] as its prompt and generates only
#      the final answer — the mind's CoT is in the prompt, not the response.
#   3. Reward is computed only on the face model's answer tokens (no classmate
#      reward term); standard GRPO advantage is used.
#   4. compute_reward() replaces compute_main_classmate_reward().
#   5. _generate_classmate_continuations() is replaced by two new methods:
#        _sample_mind_cot()      – calls frozen mind worker
#        _build_face_gen_batch() – prepends mind CoT to face's prompt tokens
#
# Naming convention:
#   "mind"  = frozen model that generates CoT  (was: classmate)
#   "face"  = trainable model that generates answer  (was: main/actor)
# ── [END MIND-FACE] ───────────────────────────────────────────────────────────

import json
import os
import uuid
from collections import defaultdict
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
# ── [MIND-FACE: CHANGED] use compute_reward (not compute_main_classmate_reward) ─
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
# ── [END MIND-FACE: CHANGED] ──────────────────────────────────────────────────
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.reward_score.BaseVerifier import get_verifier
from verl.workers.reward_model.backend_factory import build_monitor_backend, \
    build_eval_llm_judge_backend
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
# ── [MIND-FACE] reuse ClassmateWorker / ClassmateWorkerConfig as the mind model ─
from verl.workers.classmate_workers import ClassmateWorkerConfig
# ── [END MIND-FACE] ───────────────────────────────────────────────────────────


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)
    current_kl = torch.mean(current_kl, dim=0).item()

    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
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
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
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
    else:
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


# ── [MIND-FACE: CHANGED] class renamed MindFaceRayPPOTrainer ──────────────────
class MindFaceRayPPOTrainer:
    """Mind-face baseline trainer.

    Training loop:
      1. Frozen mind model generates CoT from the original prompt.
      2. Face (trainable) model receives [prompt + mind_CoT] and generates answer.
      3. Reward is computed on face's answer tokens only.
      4. GRPO advantage and policy update applied to face model.

    The ClassmateWorker infrastructure is reused for the mind model; it is
    accessed via self.mind_workers (instead of self.classmate_workers).
    """
# ── [END MIND-FACE: CHANGED] ──────────────────────────────────────────────────

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

        # ── [MIND-FACE] same worker config params as classmate trainer ─────────
        # The "classmate" model here acts as the frozen "mind" model.
        classmate_cot_reward_configs=None,
        classmate_batch_size=None,
        classmate_free_cache_engine=None,
        classmate_use_vllm=None,
        classmate_num_return_sequences=None,
        classmate_generation_configs=None,
        classmate_vllm_configs=None,
        seed=None,
        enable_thinking=None,
        think_start_str=None,
        think_end_str=None,
        classmate_think_start_str=None,
        classmate_think_end_str=None,
        # ── [END MIND-FACE] ────────────────────────────────────────────────────
    ):
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

        # ── [MIND-FACE] store mind-worker configs (reusing classmate param names) ─
        self.classmate_cot_reward_configs = classmate_cot_reward_configs
        self.classmate_batch_size = classmate_batch_size
        self.classmate_free_cache_engine = classmate_free_cache_engine
        self.classmate_use_vllm = classmate_use_vllm
        self.classmate_generation_configs = classmate_generation_configs
        self.classmate_num_return_sequences = classmate_num_return_sequences
        # ── [END MIND-FACE] ────────────────────────────────────────────────────

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

        if classmate_use_vllm:
            self.classmate_generation_configs.n = classmate_num_return_sequences
            self.classmate_generation_configs.seed = seed
            self.classmate_generation_configs.pop("do_sample")
        else:
            raise NotImplementedError("Only vLLM-based mind model is supported now.")
        OmegaConf.set_struct(self.classmate_generation_configs, True)

        self.classmate_vllm_configs = classmate_vllm_configs

        self.rng = random.Random(seed)
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        assert enable_thinking is not None, "Specify enable_thinking when initializing MindFaceRayPPOTrainer."
        self.enable_thinking = enable_thinking
        self.think_start_str = think_start_str
        self.think_end_str = think_end_str
        # ── [MIND-FACE] mind_think_start/end_str for the frozen mind model ─────
        self.mind_think_start_str = classmate_think_start_str
        self.mind_think_end_str = classmate_think_end_str
        # ── [END MIND-FACE] ────────────────────────────────────────────────────

        self.monitor_backend = build_monitor_backend(config.reward_model)
        self.eval_llm_judge_backend = build_eval_llm_judge_backend(config.reward_model)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
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

        val_batch_size = self.config.data.val_batch_size
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
            print(f"Warning: Could not set total_training_steps in config. Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
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
        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    # ── [MIND-FACE: NEW] ──────────────────────────────────────────────────────
    # These two methods replace _prepare_classmate_batch() and
    # _generate_classmate_continuations() from the classmate trainer.
    #
    # _sample_mind_cot():      calls the frozen mind worker to produce CoT text
    #                          from the raw prompt only (no face response prepended).
    # _build_face_gen_batch(): inserts mind's CoT into the face model's prompt so
    #                          the face only needs to generate the final answer.
    # ── [END MIND-FACE: NEW] ─────────────────────────────────────────────────

    def _sample_mind_cot(self, gen_batch: DataProto) -> np.ndarray:
        """Call the frozen mind model to generate CoT from each raw prompt.

        Returns an ndarray of shape (bsz,) with the mind's full generated text
        (everything after think_start_str, i.e. "<CoT text></think>\\nAnswer...").
        The trainer later extracts the CoT part (before think_end_str).
        """
        mind_input = DataProto(
            batch=None,
            non_tensor_batch={"raw_prompts": gen_batch.non_tensor_batch["raw_prompt"]},
        )
        mind_input.meta_info = {
            "mind_think_start_str": self.mind_think_start_str,
            "mind_think_end_str": self.mind_think_end_str,
            # Pass classmate_think strings for API compatibility with the worker
            "classmate_think_start_str": self.mind_think_start_str,
            "classmate_think_end_str": self.mind_think_end_str,
        }

        bsz = len(gen_batch)
        # Use the first (and typically only) mind worker
        mind_worker = self.mind_workers[0]
        size_divisor = getattr(mind_worker, "world_size", 1)

        if size_divisor > 1:
            mind_input_padded, pad_size = pad_dataproto_to_divisor(mind_input, size_divisor, pad_val=None)
            fut = mind_worker.generate_mind_cot_from_prompt(data=mind_input_padded)
        else:
            fut = mind_worker.generate_mind_cot_from_prompt(data=mind_input)
            pad_size = 0

        result = fut.get()
        # Unpad and return mind CoT texts
        mind_outputs = result.non_tensor_batch["mind_outputs"][:bsz]  # (bsz,)
        print(f"🧠 Sampled mind CoT for {bsz} prompts (pad_size={pad_size})")
        return mind_outputs

    def _build_face_gen_batch(self, gen_batch: DataProto, mind_cots: np.ndarray) -> DataProto:
        """Replace prompt tokens in gen_batch with [original_prompt + mind_CoT].

        The face model will generate only the answer starting from this extended prompt.
        We left-pad the new prompts to a uniform length using the face tokenizer.

        Args:
            gen_batch:  DataProto whose batch["input_ids"] holds original prompt tokens.
            mind_cots:  ndarray (bsz,) of mind-generated text (everything after think_start).
                        None entries fall back to the original prompt unchanged.
        """
        orig_input_ids = gen_batch.batch["input_ids"]       # (bsz, orig_prompt_len)
        orig_attention_mask = gen_batch.batch["attention_mask"]  # (bsz, orig_prompt_len)

        # Use face model's think tokens to wrap the mind's CoT in the face prompt.
        # (mind_think_start/end_str are only used to extract the CoT from the mind's output.)
        think_start_ids = self.tokenizer.encode(self.think_start_str, add_special_tokens=False)
        think_end_ids = self.tokenizer.encode(self.think_end_str, add_special_tokens=False)

        new_input_ids_list = []
        for i in range(len(gen_batch)):
            # Extract valid (non-padded) original prompt tokens
            valid_mask = orig_attention_mask[i].bool()
            valid_prompt_ids = orig_input_ids[i][valid_mask].tolist()

            mind_cot = mind_cots[i]
            if mind_cot is None:
                # Fall back: face generates from original prompt without mind CoT
                new_input_ids_list.append(valid_prompt_ids)
                continue

            # Extract CoT part from mind output: everything before think_end_str
            if self.mind_think_end_str and self.mind_think_end_str in mind_cot:
                cot_text = mind_cot[:mind_cot.index(self.mind_think_end_str)]
            else:
                cot_text = mind_cot

            cot_ids = self.tokenizer.encode(cot_text, add_special_tokens=False)

            # Face prompt: [original prompt tokens] + <think> + [mind CoT] + </think>
            # The face model generates the answer after seeing this full context.
            face_prompt_ids = valid_prompt_ids + think_start_ids + cot_ids + think_end_ids
            new_input_ids_list.append(face_prompt_ids)

        # Left-pad all face prompts to the same length
        max_len = max(len(ids) for ids in new_input_ids_list)
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_position_ids = []
        for prompt_ids in new_input_ids_list:
            pad_len = max_len - len(prompt_ids)
            padded_input_ids.append([pad_id] * pad_len + prompt_ids)
            padded_attention_mask.append([0] * pad_len + [1] * len(prompt_ids))
            padded_position_ids.append(list(range(max_len)))

        new_tensor_batch = TensorDict(
            {
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
                "position_ids": torch.tensor(padded_position_ids, dtype=torch.long),
            },
            batch_size=len(new_input_ids_list),
        )

        face_gen_batch = DataProto(
            batch=new_tensor_batch,
            non_tensor_batch=gen_batch.non_tensor_batch,
            meta_info=gen_batch.meta_info,
        )
        return face_gen_batch
    # ── [END MIND-FACE: NEW] ─────────────────────────────────────────────────

    def _validate(self):
        """Validation loop.

        Mind-face version:
          1. Mind generates CoT from each validation prompt.
          2. Face receives [prompt + mind_CoT] and generates the answer.
          3. Answer is scored by val_reward_fn (MindFaceRewardManager).
        """
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_uids = []

        monitor_score_dictionary: dict[str, dict] = {}

        print("🧠🧠🧠 Start validating (mind-face)")
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            input_ids = test_batch.batch["input_ids"]
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
                "is_eval": True,
                "global_steps": self.global_steps,
            }

            # ── [MIND-FACE: NEW] ───────────────────────────────────────────────
            # Step 1: Frozen mind model generates CoT from validation prompts.
            print("🧠 [validate] Sampling CoT from frozen mind model...")
            val_mind_cots = self._sample_mind_cot(test_gen_batch)  # (bsz,)

            # Step 2: Build face gen batch - [prompt + mind_CoT] as face's input.
            face_test_gen_batch = self._build_face_gen_batch(test_gen_batch, val_mind_cots)
            # ── [END MIND-FACE: NEW] ───────────────────────────────────────────

            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            # Pad for DP divisibility, generate, unpad
            face_test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(face_test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(face_test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(face_test_gen_batch_padded)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("🧠 [validate] Face model generation done")

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.non_tensor_batch["decoded_responses"] = np.array(
                self.tokenizer.batch_decode(test_batch.batch["responses"], skip_special_tokens=True)
            )

            # ── [MIND-FACE] CoT monitor: mind's CoT, face's answer ─────────────
            # For monitoring we use mind's CoT (not face's, since face only
            # generates the answer).  The face's decoded answer IS the output.
            data_sources_batch = test_batch.non_tensor_batch.get("data_source")
            data_source_key = str(data_sources_batch[0])

            item_raw_questions, item_final_outputs, item_final_output_ids = [], [], []
            for item, face_answer in zip(test_batch, test_batch.non_tensor_batch["decoded_responses"]):
                item_raw_questions.append(item.non_tensor_batch["reward_model"]["raw_question_for_monitor"])
                item_final_outputs.append(face_answer)
                item_final_output_ids.append(
                    self.tokenizer.encode(face_answer, add_special_tokens=False) if face_answer else []
                )

            # Mind CoTs (repeated to match val_n; one per item in the repeated batch)
            item_mind_cots = list(val_mind_cots)

            # Monitor using mind's CoT (monitors sycophantic reasoning in the CoT)
            if data_source_key == "general_reward":
                all_behavior_judge_scores = []
                for behavior_key in ["sycophancy_only", "confidence_only", "longer_response"]:
                    behavior_verifier = get_verifier(
                        data_source=behavior_key, max_new_tokens=self.config.data.max_response_length,
                    )
                    monitor_prompts = [
                        behavior_verifier.create_monitor_message({
                            "raw_question_for_monitor": rq,
                            "truncated_main_CoT": mc,
                            "main_response": mc,
                        })
                        for rq, mc in zip(item_raw_questions, item_mind_cots)
                    ]
                    behavior_monitor_scores = []
                    for raw_out in self.monitor_backend.run_batch(monitor_prompts):
                        if raw_out is None:
                            behavior_monitor_scores.append(behavior_verifier.invalid_score)
                        else:
                            score, _ = behavior_verifier.parse_monitor_output(raw_out)
                            behavior_monitor_scores.append(score)

                    judge_inputs = [
                        self.eval_llm_judge_backend.prepare_input(behavior_verifier, rq, fo)
                        for rq, fo in zip(item_raw_questions, item_final_outputs)
                    ]
                    behavior_judge_scores = []
                    for raw_out, final_ids in zip(self.eval_llm_judge_backend.run_batch(judge_inputs), item_final_output_ids):
                        judge_score, _ = behavior_verifier.parse_llm_judge_output(
                            raw_out, continuation_token_ids=final_ids
                        )
                        behavior_judge_scores.append(judge_score)

                    entry = monitor_score_dictionary.setdefault(behavior_key, {"monitor_score": [], "llm_judge_score": []})
                    entry["monitor_score"].extend(behavior_monitor_scores)
                    entry["llm_judge_score"].extend(behavior_judge_scores)
                    all_behavior_judge_scores.append(behavior_judge_scores)

                n_items = len(item_mind_cots)

                if self.val_reward_fn is None:
                    raise ValueError("val_reward_fn must be provided for validation.")
                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                actual_scores = reward_tensor.sum(-1).cpu().tolist()

                already_print_count = 0
                for i, (raw_q, mind_cot, final_out, actual_score) in enumerate(zip(
                    item_raw_questions, item_mind_cots, item_final_outputs, actual_scores
                )):
                    if already_print_count >= 1:
                        break
                    already_print_count += 1
                    print("🧠[val raw_question]", raw_q)
                    print("🧠[val mind_cot]", mind_cot)
                    print("🧠[val face_answer]", final_out)
                    print("🧠[val actual_score]", actual_score)

                sample_scores.extend(actual_scores)
                reward_extra_infos_dict["reward"].extend(actual_scores)
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)
                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["general_reward"] * n_items))
            else:
                # Non-general_reward: monitor mind's CoT, score face's answer
                monitor_scores = []
                monitor_explanations = []
                verifier = get_verifier(data_source=data_source_key, max_new_tokens=self.config.data.max_response_length)
                monitor_prompts = [
                    verifier.create_monitor_message({
                        "raw_question_for_monitor": rq,
                        "truncated_main_CoT": mc,
                        "main_response": mc,
                    })
                    for rq, mc in zip(item_raw_questions, item_mind_cots)
                ]
                for raw_out in self.monitor_backend.run_batch(monitor_prompts):
                    if raw_out is None:
                        monitor_scores.append(verifier.invalid_score)
                        monitor_explanations.append(None)
                    else:
                        score, explanation = verifier.parse_monitor_output(raw_out)
                        monitor_scores.append(score)
                        monitor_explanations.append(explanation)

                # Store mind CoT for debug / reward manager context
                test_batch.non_tensor_batch["truncated_main_cot"] = np.array(item_mind_cots)
                test_batch.non_tensor_batch["monitor_score"] = np.array(monitor_scores)
                test_batch.non_tensor_batch["monitor_explanations"] = np.array(monitor_explanations, dtype=object)
                test_batch.non_tensor_batch["monitor_reward_type"] = np.array([verifier.reward_type] * len(item_mind_cots))

                if self.val_reward_fn is None:
                    raise ValueError("val_reward_fn must be provided for validation.")
                result = self.val_reward_fn(test_batch, return_dict=True)

                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_extra_infos_dict["reward"].extend(scores)
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)

                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

                entry = monitor_score_dictionary.setdefault(data_source_key, {"monitor_score": [], "llm_judge_score": []})
                entry["monitor_score"].extend(monitor_scores)
                entry["llm_judge_score"].extend(reward_tensor.sum(-1).cpu().tolist())
            # ── [END MIND-FACE] ────────────────────────────────────────────────

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

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
            core_reward_vars = {"acc", "face_model_reward", "reward"}

            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
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

        for ds_key, score_dict in monitor_score_dictionary.items():
            if not score_dict["monitor_score"]:
                continue
            verifier = get_verifier(data_source=ds_key, max_new_tokens=self.config.data.max_response_length)
            monitor_scores = np.array(score_dict["monitor_score"])
            llm_judge_scores = np.array(score_dict["llm_judge_score"])

            monitor_valid_mask = monitor_scores != verifier.invalid_score
            judge_valid_mask = llm_judge_scores != verifier.invalid_score
            valid_mask = monitor_valid_mask & judge_valid_mask

            monitor_metrics = verifier.compute_metrics(
                predictions=monitor_scores[valid_mask].tolist(),
                ground_truths=llm_judge_scores[valid_mask].tolist(),
            )
            pfx = f"val-core/monitor/{ds_key}"
            metric_dict[f"{pfx}/total_monitored_entries"] = int(np.sum(valid_mask))
            metric_dict[f"{pfx}/total_valid_monitor_entries"] = int(np.sum(monitor_valid_mask))
            metric_dict[f"{pfx}/total_valid_judge_entries"] = int(np.sum(judge_valid_mask))
            for metric_name, metric_val in monitor_metrics.items():
                metric_dict[f"{pfx}/{metric_name}"] = float(metric_val)

        print(f"🧠 Validation metrics: {metric_dict}")
        print("🧠🧠🧠 End validating (mind-face)")
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend."""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

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

        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # ── [MIND-FACE: CHANGED] create mind model workers (using ClassmateWorker) ─
        model_paths = self.classmate_cot_reward_configs.classmate_model_name_or_path_list
        for model_idx, model_path in enumerate(model_paths):
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ClassmateWorker)

            mind_config = ClassmateWorkerConfig(
                model_name_or_path=model_path,
                model_index=model_idx,
                classmate_batch_size=self.classmate_batch_size,
                classmate_free_cache_engine=self.classmate_free_cache_engine,
                classmate_use_vllm=self.classmate_use_vllm,
                max_tokens=self.classmate_generation_configs.max_tokens,
                generation_config=self.classmate_generation_configs,
                vllm_config=self.classmate_vllm_configs,
                enable_thinking=self.enable_thinking,
                main_model_max_tokens=self.config.data.max_response_length,
            )

            mind_role_name = f"{str(Role.ClassmateWorker)}_{model_idx}"
            mind_worker_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ClassmateWorker],
                config=mind_config,
                role=mind_role_name,
            )
            self.resource_pool_to_cls[resource_pool][mind_role_name] = mind_worker_cls
        # ── [END MIND-FACE: CHANGED] ───────────────────────────────────────────

        if Role.ScoringJudgeWorker in self.role_worker_mapping:
            from verl.workers.reward_model.scoring_judge_worker import ScoringJudgeWorkerConfig
            scoring_judge_config = ScoringJudgeWorkerConfig(
                model_name_or_path=self.config.reward_model.llm_judge_model_name,
                torch_dtype=ScoringJudgeWorkerConfig.dtype_from_str(
                    self.config.reward_model.get("llm_judge_backend_dtype")
                ),
            )
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ScoringJudgeWorker)
            scoring_judge_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ScoringJudgeWorker],
                config=scoring_judge_config,
                role=str(Role.ScoringJudgeWorker),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ScoringJudgeWorker)] = scoring_judge_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                )
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
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # ── [MIND-FACE: CHANGED] initialize mind workers (self.mind_workers) ─────
        self.mind_workers = []
        model_paths = self.classmate_cot_reward_configs.classmate_model_name_or_path_list
        for model_idx, model_path in enumerate(model_paths):
            mind_role_name = f"{str(Role.ClassmateWorker)}_{model_idx}"
            mind_wg = all_wg[mind_role_name]
            mind_wg.init_model()
            self.mind_workers.append(mind_wg)
        print(f"🧠 Initialized {len(self.mind_workers)} frozen mind model worker(s)")
        # ── [END MIND-FACE: CHANGED] ────────────────────────────────────────────

        self.scoring_judge_wg = None
        if Role.ScoringJudgeWorker in self.role_worker_mapping:
            from verl.workers.reward_model.scoring_judge_worker_backend import ScoringJudgeWorkerBackend
            self.scoring_judge_wg = all_wg[str(Role.ScoringJudgeWorker)]
            self.scoring_judge_wg.init_model()
            for fn in [self.reward_fn, self.val_reward_fn]:
                if fn is not None and isinstance(getattr(fn, "llm_judge_backend", None), ScoringJudgeWorkerBackend):
                    fn.llm_judge_backend.set_worker_group(self.scoring_judge_wg)
            print("Initialized ScoringJudgeWorker and wired to reward managers")

        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager
            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

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

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str)
                assert "global_step_" in self.config.trainer.resume_from_path
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size: (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def compute_rollout_importance_weights_and_add_to_batch(self, batch: DataProto) -> tuple[DataProto, dict]:
        if self.config.algorithm.rollout_is_threshold is not None and "rollout_log_probs" in batch.batch:
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

            apply_weights = self.config.algorithm.get("rollout_is", False)
            assert not apply_weights, "actual importance ratio is calculated in the vanilla loss fn"

            if apply_weights:
                batch = batch.union(rollout_is_weights)

            return batch, rollout_is_metrics

        return batch, {}

    def fit(self):
        """Training loop.

        Mind-face order (differs from classmate trainer):
          1. [MIND] Frozen mind model generates CoT from each prompt.
          2. [FACE] Trainable face model receives [prompt + mind_CoT] and generates answer.
          3. Reward computed on face's answer tokens (no classmate term).
          4. Standard GRPO advantage + face model update.
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
        self._load_checkpoint()

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

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

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

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):

                    # ── [MIND-FACE: NEW] ─────────────────────────────────────────
                    # Step 1 (MIND): Frozen mind model generates CoT from each prompt.
                    #   - Called on gen_batch BEFORE repeat so each unique prompt gets
                    #     one mind CoT sample. Mind CoTs are then broadcast across the
                    #     n rollout repeats.
                    #   - Uses generate_mind_cot_from_prompt() added to ClassmateWorker.
                    with marked_timer("mind_cot", timing_raw, color="magenta"):
                        mind_cots = self._sample_mind_cot(gen_batch)  # (bsz,)

                    # Repeat gen_batch for n rollout samples, then repeat mind CoTs to
                    # match (interleave=True: each prompt repeated n times consecutively).
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )
                    mind_cots_repeated = np.repeat(mind_cots, self.config.actor_rollout_ref.rollout.n, axis=0)

                    # Step 2 (FACE): Build face gen batch – prepend mind CoT to prompt.
                    #   Face model sees [prompt + <think> + mind_CoT + </think>] and
                    #   generates only the final answer.
                    face_gen_batch = self._build_face_gen_batch(gen_batch_output, mind_cots_repeated)
                    # ── [END MIND-FACE: NEW] ──────────────────────────────────────

                    with marked_timer("gen", timing_raw, color="red"):
                        # Face (trainable actor) generates answer from face_gen_batch.
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(face_gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(face_gen_batch)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # Merge face outputs back into the main batch
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            # ── [MIND-FACE: CHANGED] ─────────────────────────────
                            # Use compute_reward() (not compute_main_classmate_reward)
                            # since reward comes only from the face model's answer.
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            # ── [END MIND-FACE: CHANGED] ─────────────────────────

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
                            from verl.utils.debug.metrics import calculate_debug_metrics
                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)

                        # ── [MIND-FACE: CHANGED] ──────────────────────────────────
                        # Single reward tensor on face's answer tokens (no classmate
                        # reward splitting needed).
                        batch.batch["token_level_scores"] = reward_tensor
                        # ── [END MIND-FACE: CHANGED] ─────────────────────────────

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        metrics.update(is_metrics)

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        # ── [MIND-FACE: CHANGED] ──────────────────────────────────
                        # Standard GRPO advantage on face's answer tokens.
                        # token_level_classmate_reward_mode is not used here.
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        # ── [END MIND-FACE: CHANGED] ─────────────────────────────

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            # ── [MIND-FACE] updates the face (trainable) model ────
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            # ── [END MIND-FACE] ────────────────────────────────────
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

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

                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
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

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

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

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
