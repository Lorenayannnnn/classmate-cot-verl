# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
# Entry point for mind-face baseline training.
#
# Architecture:
#   - Mind model (frozen classmate): generates CoT from the original prompt.
#   - Face model (trainable main policy): receives [prompt + mind_CoT] and
#     generates the final answer; trained with GRPO.
#
# Based on classmate_cot_main_ppo.py; uses MindFaceRayPPOTrainer and the
# "mind_face" reward manager.
# ── [END MIND-FACE] ──────────────────────────────────────────────────────────

"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo, create_rl_dataset, create_rl_sampler

# ── [MIND-FACE: CHANGED] import MindFaceRayPPOTrainer instead of ClassmateCoTRayPPOTrainer ──
from verl.trainer.ppo.mind_face_ray_trainer import MindFaceRayPPOTrainer
# ── [END MIND-FACE: CHANGED] ─────────────────────────────────────────────────

from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy, set_seed
from verl.workers.reward_model.backend_factory import build_llm_judge_backend
from verl.utils.config import validate_config


# ── [MIND-FACE: CHANGED] config_name points to mind_face_ppo_trainer ─────────
@hydra.main(config_path="config", config_name="mind_face_ppo_trainer", version_base=None)
def main(config):
    """Main entry point for mind-face PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    set_seed(config.data.seed)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(MindFaceTaskRunner))
# ── [END MIND-FACE: CHANGED] ─────────────────────────────────────────────────


# ── [MIND-FACE: CHANGED] class renamed MindFaceTaskRunner ────────────────────
class MindFaceTaskRunner(TaskRunner):
    """Ray remote class for executing distributed mind-face PPO training tasks.

    Extends TaskRunner with mind model worker setup.  The mind model reuses
    ClassmateWorker infrastructure; it is frozen during training and generates
    CoT prefixes for the face model.
    """

    def add_mind_rollout_worker(self, config):
        """Add mind (frozen CoT generator) worker using ClassmateWorker infrastructure."""
        from verl.trainer.ppo.utils import Role

        model_paths = config.reward_model.classmate_cot_reward_configs.classmate_model_name_or_path_list
        assert len(model_paths) > 0, "No mind model configured"

        from verl.workers.classmate_workers import ClassmateWorker

        self.role_worker_mapping[Role.ClassmateWorker] = ray.remote(ClassmateWorker)
        self.mapping[Role.ClassmateWorker] = "global_pool"

        print(f"[mind-face] Added mind worker role for model: {model_paths}")

    def run(self, config):
        """Execute the main mind-face PPO training workflow."""
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_worker(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)
        # ── [MIND-FACE: CHANGED] add mind worker (not classmate worker) ───────
        self.add_mind_rollout_worker(config)
        # ── [END MIND-FACE: CHANGED] ──────────────────────────────────────────
        self.add_scoring_judge_worker(config)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        llm_judge_backend = build_llm_judge_backend(config.reward_model)

        # ── [MIND-FACE] reward_fn uses "mind_face" reward manager (see config) ─
        # No classmate_cot_reward_configs needed — mind-face uses a simpler
        # single-reward path scored on the face model's answer tokens only.
        reward_kwargs = dict(
            code_api_url=config.reward_model.sandbox_fusion_url,
            enable_thinking=config.reward_model.enable_thinking,
            think_start_str=config.reward_model.get("think_start_str"),
            think_end_str=config.reward_model.get("think_end_str"),
            max_new_tokens=config.data.max_response_length,
            llm_judge_backend=llm_judge_backend,
        )
        # ── [END MIND-FACE] ────────────────────────────────────────────────────
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=10, **reward_kwargs)

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ── [MIND-FACE: CHANGED] instantiate MindFaceRayPPOTrainer ───────────
        trainer = MindFaceRayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            # Mind model config — reuses classmate_* param names for compatibility
            # with ClassmateWorker infrastructure.
            classmate_cot_reward_configs=config.reward_model.classmate_cot_reward_configs,
            classmate_batch_size=config.reward_model.classmate_batch_size,
            classmate_use_vllm=config.reward_model.classmate_use_vllm,
            classmate_num_return_sequences=config.reward_model.classmate_num_return_sequences,
            classmate_generation_configs=config.reward_model.classmate_generation_configs,
            seed=config.data.seed,
            enable_thinking=config.reward_model.enable_thinking,
            think_start_str=config.reward_model.think_start_str,
            think_end_str=config.reward_model.think_end_str,
            classmate_think_start_str=config.reward_model.get("classmate_think_start_str"),
            classmate_think_end_str=config.reward_model.get("classmate_think_end_str"),
        )
        # ── [END MIND-FACE: CHANGED] ──────────────────────────────────────────
        trainer.init_workers()
        trainer.fit()
# ── [END MIND-FACE: CHANGED] ─────────────────────────────────────────────────


if __name__ == "__main__":
    ray_temp_dir = os.path.expanduser("~/ray_tmp")
    os.makedirs(ray_temp_dir, exist_ok=True)
    os.environ["RAY_TMPDIR"] = ray_temp_dir
    main()
