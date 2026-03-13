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
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray

from verl.trainer.main_ppo import TaskRunner, run_ppo, create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.classmate_cot_ray_trainer import ClassmateCoTRayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy, set_seed
from verl.workers.reward_model.backend_factory import build_llm_judge_backend
from verl.utils.config import validate_config


@hydra.main(config_path="config", config_name="classmate_cot_ppo_trainer", version_base=None)
def main(config):
    """Main entry point for classmate CoT PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    set_seed(config.data.seed)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(ClassmateCoTTaskRunner))


class ClassmateCoTTaskRunner(TaskRunner):
    """Ray remote class for executing distributed classmate CoT PPO training tasks.

    Extends TaskRunner with classmate model sampling and reward. Only overrides
    worker setup and the run() method; all shared worker logic is inherited.
    """

    def add_classmate_rollout_worker(self, config):
        """Add classmate worker if classmate CoT reward is enabled."""
        from verl.trainer.ppo.utils import Role

        model_paths = config.reward_model.classmate_cot_reward_configs.classmate_model_name_or_path_list
        assert len(model_paths) > 0, "No classmate models configured"

        from verl.workers.classmate_workers import ClassmateWorker

        self.role_worker_mapping[Role.ClassmateWorker] = ray.remote(ClassmateWorker)
        self.mapping[Role.ClassmateWorker] = "global_pool"

        print(f"Added classmate worker role for {len(model_paths)} classmate models")

    def run(self, config):
        """Execute the main classmate CoT PPO training workflow."""
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
        self.add_classmate_rollout_worker(config)

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

        reward_kwargs = dict(
            code_api_url=config.reward_model.sandbox_fusion_url,
            classmate_cot_reward_configs=config.reward_model.classmate_cot_reward_configs,
            classmate_generation_configs=config.reward_model.classmate_generation_configs,
            enable_thinking=config.reward_model.enable_thinking,
            think_start_str=config.reward_model.get("think_start_str"),
            think_end_str=config.reward_model.get("think_end_str"),
            max_new_tokens=config.data.max_response_length,
            llm_judge_backend=llm_judge_backend,
        )
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

        trainer = ClassmateCoTRayPPOTrainer(
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
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    ray_temp_dir = os.path.expanduser("~/ray_tmp")
    os.makedirs(ray_temp_dir, exist_ok=True)
    os.environ["RAY_TMPDIR"] = ray_temp_dir
    main()