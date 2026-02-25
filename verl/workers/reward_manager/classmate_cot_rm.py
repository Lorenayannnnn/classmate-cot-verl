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
import time
from collections import defaultdict
from typing import Any
from concurrent.futures import Future
import torch
import ray
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl import DataProto
from verl.utils.reward_score.cot_monitor.monitor import parse_out_main_cot_output
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
            enable_thinking=False,
            llm_judge_model=None, llm_judge_timeout=None, llm_judge_max_tokens=None,
            llm_judge_max_context_length=None, llm_judge_temperature=None, seed=None,
            classmate_cot_reward_configs=None,
            classmate_generation_configs=None,
            max_workers=8,
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
            max_workers: Maximum number of worker threads for parallel processing.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.max_workers = max_workers

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
        #             )å
        #         ),
        #     }

        self.enable_thinking = enable_thinking

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
        # consistency_reward_tensor = None
        # if self.classmate_cot_reward_configs.add_consistency_reward:
        #     consistency_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # print("🐛🐛🐛 data", data)
        # print("🐛🐛🐛 classmate_outputs", data.non_tensor_batch["classmate_outputs"].shape)

        def _as_future(result):
            future = Future()
            future.set_result(result)
            return future

        think_open_id  = self.tokenizer.convert_tokens_to_ids("<think>")
        think_close_id = self.tokenizer.convert_tokens_to_ids("</think>")

        item_contexts = []
        main_score_futures = []

        # print("🐛🐛🐛 Start scoring main classmate outputs")

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

            # Extract things after <think>...</think> as output if enable_thinking is True
            if self.enable_thinking:
                main_cot, main_output, _, _ = parse_out_main_cot_output(
                    response_str, valid_response_ids, self.tokenizer, think_open_id, think_close_id
                )
            else:
                main_cot = response_str
                main_output = response_str

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores

            extra_info["reward_model"] = data_item.non_tensor_batch["reward_model"]

            # Submit main model reward computation
            main_score_futures.append(
                _as_future(
                    self.compute_score(
                        data_source=data_source,
                        # solution_str=response_str,
                        solution_str=main_output,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        return_dict=True,
                        code_api_url=self.code_api_url,
                        # llm_judge_config_dict=self.llm_judge_config_dict,
                        # code_verifier_src_to_verifier=self.code_verifier_src_to_verifier,
                    )
                )
            )

            classmate_prompts = data_item.non_tensor_batch["classmate_prompts"]
            classmate_outputs = data_item.non_tensor_batch["classmate_outputs"]     # (num_classmate_models, num_samples_per_classmate). Currently only support num_samples_per_classmate=1
            main_keep_rate = data_item.non_tensor_batch["main_keep_rates"]     # (num_classmate_models, num_samples_per_classmate). Currently only support num_samples_per_classmate=1
            classmate_ground_truth = data_item.non_tensor_batch["all_other_prompt_gts"] if "all_other_prompt_gts" in data_item.non_tensor_batch else ground_truth

            item_contexts.append(
                {
                    "index": i,
                    "data_item": data_item,
                    "raw_prompt": raw_prompt,
                    "raw_prompt_str": raw_prompt_str,
                    "raw_response_str": raw_response_str,
                    "response_str": response_str,
                    "main_cot": main_cot,
                    "main_output": main_output,
                    "ground_truth": ground_truth,
                    "data_source": data_source,
                    "extra_info": extra_info,
                    "valid_response_length": valid_response_length,
                    "classmate_prompts": classmate_prompts,
                    "classmate_outputs": classmate_outputs,
                    "main_keep_rate": main_keep_rate,
                    "classmate_ground_truth": classmate_ground_truth,
                }
            )

        # print("🐛🐛🐛 Finish submitting main outputs")

        # from tqdm import tqdm
        # main_p_bar = tqdm(total=len(item_contexts), desc="Scoring main model outputs")

        # Second loop: resolve main rewards and submit classmate reward computations
        for ctx, main_score_future in zip(item_contexts, main_score_futures):
            actor_score = main_score_future.result()
            # main_p_bar.update(1)
            extracted_solution = actor_score.pop("extracted_solution", None) if isinstance(actor_score, dict) else None

            if isinstance(actor_score, dict):
                base_reward = actor_score["score"]
                # Store the information including original reward
                for key, value in actor_score.items():
                    reward_extra_info[f"main_model_{key}"].append(value)
            else:
                base_reward = actor_score

            reward_extra_info["main_model_reward"].append(base_reward)

            ctx["actor_score"] = actor_score
            ctx["extracted_solution"] = extracted_solution
            ctx["base_reward"] = base_reward

            classmate_outputs = ctx["classmate_outputs"]
            classmate_model_weights = [1.0 / len(classmate_outputs)] * len(classmate_outputs)
            classmate_future_groups = []

            for classmate_output in classmate_outputs:
                sample_futures = []
                for classmate_output_sample in classmate_output:
                    sample_futures.append(
                        _as_future(
                            self.compute_score(
                                data_source=ctx["data_source"],
                                solution_str=classmate_output_sample,
                                ground_truth=ctx["classmate_ground_truth"],
                                extra_info=ctx["extra_info"],
                                method="flexible",      # this doesn't matter; just a dummy placeholder
                                return_dict=True,
                                code_api_url=self.code_api_url,
                                # llm_judge_config_dict=self.llm_judge_config_dict,
                                # code_verifier_src_to_verifier=self.code_verifier_src_to_verifier
                            )
                        )
                    )
                classmate_future_groups.append(sample_futures)

            ctx["classmate_model_weights"] = classmate_model_weights
            ctx["classmate_future_groups"] = classmate_future_groups

            main_reward_tensor[ctx["index"], ctx["valid_response_length"] - 1] = base_reward

        # classmate_p_bar = tqdm(total=len(item_contexts), desc="Scoring classmate model outputs")

        # Third loop: resolve classmate rewards
        for ctx in item_contexts:
            total_classmate_reward_dict = {}
            extracted_classmate_sol = []

            classmate_outputs = ctx["classmate_outputs"]
            classmate_model_weights = ctx["classmate_model_weights"]
            classmate_future_groups = ctx["classmate_future_groups"]

            for classmate_idx, (classmate_output, classmate_futures) in enumerate(
                zip(classmate_outputs, classmate_future_groups)
            ):
                tmp_total_classmate_reward_dict = {}
                tmp_extracted_classmate_sol_list = []
                for classmate_future in classmate_futures:
                    tmp_classmate_result = classmate_future.result()
                    tmp_extracted_classmate_sol = tmp_classmate_result.pop("extracted_solution", None)
                    tmp_extracted_classmate_sol_list.append(tmp_extracted_classmate_sol)

                    for k, v in tmp_classmate_result.items():       # Should only have "score"
                        if k not in tmp_total_classmate_reward_dict:
                            tmp_total_classmate_reward_dict[k] = 0.0
                        tmp_total_classmate_reward_dict[k] += v

                extracted_classmate_sol.append(tmp_extracted_classmate_sol_list)

                for k, v in tmp_total_classmate_reward_dict.items():
                    if k not in total_classmate_reward_dict:
                        total_classmate_reward_dict[k] = 0.0
                    total_classmate_reward_dict[k] += v / len(classmate_output) * classmate_model_weights[classmate_idx]

            base_reward = ctx["base_reward"]
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

            reward_extra_info[f"classmate_response_length"].append(ctx["data_item"].non_tensor_batch["classmate_response_length"])
            reward_extra_info[f"classmate_max_tokens_len"].append(ctx["data_item"].non_tensor_batch["classmate_max_tokens_len"])

            # Note: this is only for logging; token-level reward might be different
            final_reward = base_reward + weighted_classmate_reward
            reward_extra_info["final_reward"].append(final_reward)

            classmate_reward_tensor[ctx["index"], ctx["valid_response_length"] - 1] = weighted_classmate_reward

            data_source = ctx["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # classmate_p_bar.update(1)

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("🐛[raw_prompt]", ctx["raw_prompt"])
                print("🐛[main_prompt]", ctx["raw_prompt_str"])
                print("🐛[main_response]", ctx["raw_response_str"])
                if self.enable_thinking:
                    print("🐛[main_cot]", ctx["main_cot"])
                    print("🐛[main_output]", ctx["main_output"])
                ground_truth = ctx["ground_truth"]
                actor_score = ctx["actor_score"]
                extracted_solution = ctx["extracted_solution"]
                if ground_truth is None:
                    # for open-ended generation
                    ground_truth = actor_score.get("score", None) if isinstance(actor_score, dict) else None
                    print("🐛[llm judge score]", ground_truth)
                    if isinstance(actor_score, dict):
                        print("🐛[llm judge explanation]", actor_score.get("judge_explanation", None))
                else:
                    # if extracted_solution is not None:
                    print("🐛[main_extracted]", extracted_solution)
                    print("🐛[ground_truth]", ground_truth)
                    print("🐛[main_reward]", base_reward)
                import numpy as np
                print("🐛[classmate_prompt]", ctx["classmate_prompts"][0])   # 0th classmate model
                print("🐛[main_keep_rate]", np.array2string(ctx["main_keep_rate"], precision=3))   # 0th classmate model, 0th sample
                print("🐛[classmate_output]", ctx["classmate_outputs"][0][0])     # 0th classmate model
                print("🐛[extracted_classmate_sol]", extracted_classmate_sol[0][0])   # 0th classmate model, 0th sample
                if ground_truth is not None:
                    print("🐛[classmate_ground_truth (same as main ground truth)]", ground_truth)
                print("🐛[classmate_reward]", classmate_reward)
                print("🐛[weighted_classmate_reward]", weighted_classmate_reward)
                print("🐛[final_reward]", final_reward)
                if "classmate_outputs" in ctx["data_item"].non_tensor_batch:
                    print(f"[num_classmate_outputs]", len(ctx["data_item"].non_tensor_batch["classmate_outputs"]))

                if "monitor_score" in ctx["data_item"].non_tensor_batch:
                    print("🐛[truncated_main_cot]", ctx["data_item"].non_tensor_batch["truncated_main_cot"])
                    print("🐛[monitor_score]", ctx["data_item"].non_tensor_batch["monitor_score"])
                    print("🐛[monitor_explanation]", ctx["data_item"].non_tensor_batch["monitor_explanations"])
                    monitor_reward_type = ctx["data_item"].non_tensor_batch["monitor_reward_type"]
                    monitor_score_value = ctx["data_item"].non_tensor_batch["monitor_score"]
                    if monitor_reward_type == "binary":
                        main_monitor_consistent = (base_reward == 1) == (monitor_score_value == 1)
                        print("🐛[main_monitor_consistent]", main_monitor_consistent)
                    else:
                        try:
                            main_monitor_consistent = (float(base_reward) - float(monitor_score_value)) ** 2
                        except (TypeError, ValueError):
                            main_monitor_consistent = None
                        print("🐛[main_monitor_mse_diff]", main_monitor_consistent)


        if return_dict:
            result = {
                "main_reward_tensor": main_reward_tensor,
                "classmate_reward_tensor": classmate_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
            # if self.classmate_cot_reward_configs.add_consistency_reward:
            #     result["consistency_reward_tensor"] = consistency_reward_tensor
            return result
        else:
            return main_reward_tensor, classmate_reward_tensor, None  # consistency_reward_tensor


