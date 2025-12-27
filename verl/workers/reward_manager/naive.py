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
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.olmo_verifiers import verify_math, CodeVerifier, CodeVerifierConfig, verify_ifeval
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source",
                 code_api_url=None, llm_judge_model=None, llm_judge_timeout=None, llm_judge_max_tokens=None,
                 llm_judge_max_context_length=None, llm_judge_temperature=None, seed=None):
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        # TODO Modify
        self.code_api_url = code_api_url
        # if llm_judge_model is not None:     # TODO may remove later; not used by olmo2
        #     self.llm_judge_config_dict = {
        #         "llm_judge_model": llm_judge_model,
        #         "llm_judge_timeout": llm_judge_timeout,
        #         "llm_judge_max_tokens": llm_judge_max_tokens,
        #         "llm_judge_max_context_length": llm_judge_max_context_length,
        #         "llm_judge_temperature": llm_judge_temperature,
        #         "seed": seed,
        #     }
        # else:
        #     self.llm_judge_config_dict = None
        #
        # if code_api_url is not None:        # TODO may remove later; not used by olmo2
        #     self.code_verifier_src_to_verifier = {
        #         "code": CodeVerifier(
        #             verifier_config=CodeVerifierConfig(
        #                 code_api_url=code_api_url + "/test_program",
        #                 code_max_execution_time=1.0,
        #                 code_pass_rate_reward_threshold=0.99,
        #                 code_apply_perf_penalty=False,
        #             )
        #         ),
        #         "code_stdio": CodeVerifier(
        #             verifier_config=CodeVerifierConfig(
        #                 code_api_url=code_api_url + "/test_program_stdio",
        #                 code_max_execution_time=1.0,
        #                 code_pass_rate_reward_threshold=0.99,
        #                 code_apply_perf_penalty=False,
        #             )
        #         ),
        #     }
        # else:
        #     self.code_verifier_src_to_verifier = None

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

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

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            raw_prompt = data_item.non_tensor_batch["raw_prompt"]
            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            # TODO modify
            extra_info["reward_model"] = data_item.non_tensor_batch["reward_model"]

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                return_dict=True,
                code_api_url=self.code_api_url,
                # llm_judge_config_dict=self.llm_judge_config_dict,
                # code_verifier_src_to_verifier=self.code_verifier_src_to_verifier
            )

            # Debug
            extracted_sol = score.pop("extracted_solution", None)

            # for code_contests_modify_code
            # score = {
            #     "cot": reasoning_str,
            #     "generated_test_pass_correct_sol": 0.0,
            #     "generated_test_pass_incorrect_sol": 0.0,
            #     "generated_code_pass_gt_test": 0.0,
            #     "generated_code_pass_generated_test": 0.0,
            # }

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    if key == "score":
                        reward_extra_info["main_model_reward"].append(value)
                    else:
                        reward_extra_info[f"main_model_{key}"].append(value)
            else:
                reward = score

            # reward_extra_info["main_model_reward"].append(reward)     # Modify: Temporarily added for plotting main model's reward of baseline and one trained with classmate in the same figure
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("🐛[raw_prompt]", raw_prompt)
                print("🐛[main_prompt]", prompt_str)
                print("🐛[main_response]", response_str)
                if extracted_sol is not None:
                    print("🐛[main_extracted]", extracted_sol)
                print("🐛[ground_truth]", ground_truth)
                # print("🐛 [prompt w/ special tok]", self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False))
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"🐛[{key}]", value)
                else:
                    print("[main_reward]", score)
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
