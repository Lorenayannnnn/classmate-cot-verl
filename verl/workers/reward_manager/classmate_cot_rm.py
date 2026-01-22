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
from verl.utils.reward_score.olmo_verifiers import CodeVerifier, CodeVerifierConfig
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
            code_api_url=None,
            llm_judge_model=None, llm_judge_timeout=None, llm_judge_max_tokens=None,
            llm_judge_max_context_length=None, llm_judge_temperature=None, seed=None,
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

        self.code_api_url = code_api_url
        # if llm_judge_model is not None:     # TODO may remove later; not used by olmo2
        #     self.llm_judge_config_dict = {
        #         "llm_judge_model": llm_judge_model,
        #         "llm_judge_timeout": llm_judge_timeout,
        #         "llm_judge_max_tokens": llm_judge_max_tokens,
        #         "llm_judge_max_context_length": llm_judge_max_context_length,
        #         "llm_judge_temperature": llm_judge_temperature,
        #         "seed": seed,     # Dropped for deepinfra
        #     }

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

        main_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        classmate_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # print("🐛🐛🐛 data", data)
        # print("🐛🐛🐛 classmate_outputs", data.non_tensor_batch["classmate_outputs"].shape)

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
            raw_prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            raw_response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            extra_info["reward_model"] = data_item.non_tensor_batch["reward_model"]

            # Compute base actor reward
            actor_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                return_dict=True,
                code_api_url=self.code_api_url,
                # llm_judge_config_dict=self.llm_judge_config_dict,
                # code_verifier_src_to_verifier=self.code_verifier_src_to_verifier,
            )
            # Debugging
            extracted_solution = actor_score.pop("extracted_solution", None)

            if isinstance(actor_score, dict):
                base_reward = actor_score["score"]
                # Store the information including original reward
                for key, value in actor_score.items():
                    reward_extra_info[f"main_model_{key}"].append(value)
            else:
                base_reward = actor_score

            reward_extra_info["main_model_reward"].append(base_reward)
            # -------------------- Calculate Classmate Rewards --------------------
            total_classmate_reward_dict = {}
            classmate_prompts = data_item.non_tensor_batch["classmate_prompts"]
            classmate_outputs = data_item.non_tensor_batch["classmate_outputs"]     # (num_classmate_models, num_samples_per_classmate). Currently only support num_samples_per_classmate=1
            main_keep_rate = data_item.non_tensor_batch["main_keep_rates"]     # (num_classmate_models, num_samples_per_classmate). Currently only support num_samples_per_classmate=1

            # TODO weight for each classmate model
            classmate_model_weights = [1.0 / len(classmate_outputs)] * len(classmate_outputs)

            # Debug
            extracted_classmate_sol = []
            classmate_ground_truth = data_item.non_tensor_batch["all_other_prompt_gts"]["ground_truth"] if "all_other_prompt_gts" in data_item.non_tensor_batch else ground_truth

            for classmate_idx, classmate_output in enumerate(classmate_outputs):   # Iterate over num_classmate_models
                tmp_total_classmate_reward_dict = {}
                tmp_extracted_classmate_sol_list = []
                for classmate_output_sample in classmate_output:   # Iterate over num_return_sequences
                    tmp_classmate_result = self.compute_score(
                        data_source=data_source,
                        solution_str=classmate_output_sample,
                        ground_truth=classmate_ground_truth,
                        extra_info=extra_info,
                        # TODO flexible for classmate (not strict requirement on format)
                        method="flexible",
                        return_dict=True,
                        code_api_url=self.code_api_url,
                        # llm_judge_config_dict=self.llm_judge_config_dict,
                        # code_verifier_src_to_verifier=self.code_verifier_src_to_verifier
                    )

                    # Debug
                    tmp_extracted_classmate_sol = tmp_classmate_result.pop("extracted_solution", None)
                    tmp_extracted_classmate_sol_list.append(tmp_extracted_classmate_sol)

                    for k, v in tmp_classmate_result.items():       # Should only have "score"
                        if k not in tmp_total_classmate_reward_dict:
                            tmp_total_classmate_reward_dict[k] = 0.0
                        tmp_total_classmate_reward_dict[k] += v

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

                extracted_classmate_sol.append(tmp_extracted_classmate_sol_list)

                for k, v in tmp_total_classmate_reward_dict.items():
                    if k not in total_classmate_reward_dict:
                        total_classmate_reward_dict[k] = 0.0
                    total_classmate_reward_dict[k] += v / len(classmate_output) * classmate_model_weights[classmate_idx]

            # print(f"🐛classmate_outputs: {classmate_outputs} classmate_reward: {classmate_reward}")
            # breakpoint()

            # Combine actor and classmate rewards
            # final_reward = base_reward + classmate_reward * self.classmate_reward_weight
            # final_reward = (1 - self.classmate_reward_weight) * base_reward + self.classmate_reward_weight * classmate_reward
            classmate_reward = total_classmate_reward_dict["score"]

            weighted_classmate_reward = self.classmate_reward_weight * classmate_reward
            if self.classmate_cot_reward_configs.use_classmate_main_cond == "no_classmate":
                weighted_classmate_reward = 0
            elif self.classmate_cot_reward_configs.use_classmate_main_cond == "no_classmate_when_main_incorrect":
                # Multiply 0!!! Don't set it to 0!! Or it's possible that you are setting classmate reward to 1 when main & classmate are incorrect
                weighted_classmate_reward *= 0 if base_reward <= 0 else 1        # TODO assuming RLVR binary reward: 0 classmate reward when main is incorrect
            elif self.classmate_cot_reward_configs.use_classmate_main_cond == "neg_classmate_when_main_incorrect":
                if base_reward <= 0:
                    weighted_classmate_reward = -weighted_classmate_reward

            # Update total_classmate_reward_dict to reward_extra_info
            for key, value in total_classmate_reward_dict.items():
                if key == "score":
                    reward_extra_info["classmate_reward"].append(classmate_reward)
                else:
                    reward_extra_info[f"classmate_{key}"].append(value)

            reward_extra_info[f"weighted_classmate_reward"].append(weighted_classmate_reward)
            reward_extra_info[f"classmate_response_length"].append(data_item.non_tensor_batch["classmate_response_length"])
            reward_extra_info[f"classmate_max_tokens_len"].append(data_item.non_tensor_batch["classmate_max_tokens_len"])

            # print(f"🐛🐛🐛use_classmate_main_cond", self.classmate_cot_reward_configs.use_classmate_main_cond, type(self.classmate_cot_reward_configs.use_classmate_main_cond))
            # print(f"🐛🐛🐛base_reward", base_reward, type(base_reward))
            # print(f"🐛🐛🐛classmate_reward", classmate_reward, type(classmate_reward))
            # print(f"🐛🐛🐛weighted_classmate_reward", weighted_classmate_reward, type(weighted_classmate_reward))

            # Note: this is only for logging; token-level reward might be different
            final_reward = base_reward + weighted_classmate_reward
            reward_extra_info["final_reward"].append(final_reward)

            main_reward_tensor[i, valid_response_length - 1] = base_reward
            classmate_reward_tensor[i, valid_response_length - 1] = weighted_classmate_reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("🐛[raw_prompt]", raw_prompt)
                print("🐛[main_prompt]", raw_prompt_str)
                print("🐛[main_response]", raw_response_str)
                print("🐛[ground_truth]", ground_truth)
                if extracted_solution is not None:
                    print("🐛[main_extracted]", extracted_solution)
                print("🐛[main_reward]", base_reward)
                import numpy as np
                print("🐛[classmate_prompt]", classmate_prompts[0])   # 0th classmate model
                print("🐛[main_keep_rate]", np.array2string(main_keep_rate, precision=3))   # 0th classmate model, 0th sample
                print("🐛[classmate_output]", classmate_outputs[0][0])     # 0th classmate model
                print("🐛[extracted_classmate_sol]", extracted_classmate_sol[0][0])   # 0th classmate model, 0th sample
                print("🐛[classmate_ground_truth]", classmate_ground_truth)
                print("🐛[classmate_reward]", classmate_reward)
                print("🐛[weighted_classmate_reward]", weighted_classmate_reward)
                # if isinstance(score, dict):
                #     for key, value in score.items():
                #         print(f"[{key}]", value)
                # else:
                #     print("[score]", score)
                print("🐛[final_reward]", final_reward)
                if "classmate_outputs" in data_item.non_tensor_batch:
                    print(f"[num_classmate_outputs]", len(data_item.non_tensor_batch["classmate_outputs"]))

        if return_dict:
            return {
                "main_reward_tensor": main_reward_tensor,
                "classmate_reward_tensor": classmate_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return main_reward_tensor, classmate_reward_tensor


