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

from collections import defaultdict
from typing import Any, List, Dict
import re

import torch
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("classmate_cot")
class ClassmateCoTRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(
            self, 
            tokenizer, 
            num_examine, 
            compute_score=None, 
            reward_fn_key="data_source", 
            classmate_cot_reward_configs=None, 
            classmate_generation_configs=None,
    ) -> None:
        """
        Initialize the ClassmateCoTRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
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
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        # Store configurations
        self.classmate_cot_reward_configs = classmate_cot_reward_configs
        self.classmate_generation_configs = classmate_generation_configs
        
        # Get reward weight for combining actor and classmate rewards
        self.classmate_reward_weight = self.classmate_cot_reward_configs.classmate_reward_weight
    

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards incorporating classmate chain-of-thought outputs."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # print("🐛🐛🐛 data", data)
        # print("🐛🐛🐛 classmate_outputs", data.non_tensor_batch["classmate_outputs"].shape)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # Compute base actor reward
            actor_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(actor_score, dict):
                base_reward = actor_score["score"]
                # Store the information including original reward
                for key, value in actor_score.items():
                    reward_extra_info[f"actor_{key}"].append(value)
            else:
                base_reward = actor_score
                reward_extra_info["actor_score"].append(base_reward)
            
            # -------------------- Calculate Classmate Rewards --------------------
            classmate_reward = 0.0
            classmate_outputs = data_item.non_tensor_batch["classmate_outputs"]     # (num_classmate_models, num_samples_per_classmate). Currently only support num_samples_per_classmate=1

            # TODO weight for each classmate model
            classmate_model_weights = [1.0 / len(classmate_outputs)] * len(classmate_outputs)

            for classmate_idx, classmate_output in enumerate(classmate_outputs):   # Iterate over num_classmate_models
                tmp_classmate_reward = 0.0
                for classmate_output_sample in classmate_output:   # Iterate over num_return_sequences
                    tmp_classmate_reward += self.compute_score(
                        data_source=data_source,
                        solution_str=classmate_output_sample,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        # TODO flexible for classmate (not strict requirement on format)
                        method="flexible",
                    )
                    # print(f"""
                    # 🐛 Sample data_source: {data_source}
                    # 🐛 Sample ground_truth: {ground_truth}
                    # 🐛 Sample extra_info: {extra_info}
                    # 🐛 Sample main model response: {response_str}
                    # 🐛 Sample main model reward: {base_reward}
                    # 🐛 Sample classmate response: {classmate_output_sample}
                    # 🐛 Sample tmp_classmate_reward: {tmp_classmate_reward}
                    # """)
                # Average over num_return_sequences
                tmp_classmate_reward /= len(classmate_output)
                classmate_reward += tmp_classmate_reward * classmate_model_weights[classmate_idx]

            # print(f"🐛classmate_outputs: {classmate_outputs} classmate_reward: {classmate_reward}")
            # breakpoint()

            # Combine actor and classmate rewards
            # final_reward = base_reward + classmate_reward * self.classmate_reward_weight
            # final_reward = (1 - self.classmate_reward_weight) * base_reward + self.classmate_reward_weight * classmate_reward
            final_reward = base_reward + self.classmate_reward_weight * classmate_reward
            reward_extra_info["base_reward"].append(base_reward)
            reward_extra_info["classmate_reward"].append(classmate_reward)
            reward_extra_info["final_reward"].append(final_reward)

            reward_tensor[i, valid_response_length - 1] = final_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[actor_response]", response_str)
                print("[ground_truth]", ground_truth)
                # if isinstance(score, dict):
                #     for key, value in score.items():
                #         print(f"[{key}]", value)
                # else:
                #     print("[score]", score)
                print("[base_reward]", base_reward)
                print("[classmate_reward]", classmate_reward)
                print("[final_reward]", final_reward)
                if "classmate_outputs" in data_item.non_tensor_batch:
                    print(f"[num_classmate_outputs]", len(data_item.non_tensor_batch["classmate_outputs"]))

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


