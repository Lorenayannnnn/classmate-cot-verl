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
from verl.utils.reward_score.BaseVerifier import get_verifier
from verl.utils.reward_score.monitor import parse_out_main_cot_output
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_model.judge_backend import BaseBackend


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source",
                 code_api_url=None, enable_thinking=False,
                 think_start_str=None, think_end_str=None,
                 max_new_tokens=None,
                 llm_judge_backend: BaseBackend | None = None,
                 ):
        """
        Args:
            tokenizer: Tokenizer used to decode token IDs into text.
            num_examine: Number of batches of decoded responses to print for debugging.
            compute_score: Unused; kept for interface compatibility.
            reward_fn_key: Key used to look up data_source in non_tensor_batch.
            llm_judge_backend: Backend used for judge inference. Pass a
                TinkerJudgeBackend, VLLMGenerativeJudgeBackend, or
                VLLMScoringJudgeBackend instance; constructed by the trainer from
                reward_model.llm_judge_model_name.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        self.max_new_tokens = max_new_tokens
        self.code_api_url = code_api_url
        self.enable_thinking = enable_thinking
        self.think_start_str = think_start_str
        self.think_end_str = think_end_str

        if self.enable_thinking:
            assert self.think_start_str and self.think_end_str, \
                "think_start_str and think_end_str must be set when enable_thinking=True."

        assert llm_judge_backend is not None, \
            "llm_judge_backend must be configured for non-verifiable reward."
        self.llm_judge_backend = llm_judge_backend

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """We will expand this function gradually based on the available datasets"""

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

        # ── Loop 1: decode responses, build item contexts, collect judge prompts ──
        item_contexts = []
        judge_prompts = []

        for i in range(len(data)):
            data_item = data[i]

            raw_prompt = data_item.non_tensor_batch["raw_prompt"]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.enable_thinking:
                main_cot, main_output, _, cot_end = parse_out_main_cot_output(
                    response_str, valid_response_ids, self.tokenizer,
                    self.think_start_str, self.think_end_str,
                )
                main_output_ids = valid_response_ids[cot_end:] if cot_end is not None else valid_response_ids
            else:
                main_cot = response_str
                main_output = response_str
                main_output_ids = valid_response_ids

            # ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            # ground_truth_for_llm_judge = data_item.non_tensor_batch["reward_model"].get(
            #     "ground_truth_for_llm_judge", ground_truth)
            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["num_turns"] = num_turns
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            extra_info["reward_model"] = data_item.non_tensor_batch["reward_model"]

            item_contexts.append({
                "index": i,
                "data_item": data_item,
                "raw_prompt": raw_prompt,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "main_cot": main_cot,
                "main_output": main_output,
                "main_output_ids": main_output_ids,
                # "ground_truth": ground_truth,
                # "ground_truth_for_llm_judge": ground_truth_for_llm_judge,
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length,
            })

            if main_output is None:
                judge_prompts.append(None)
            else:
                verifier = get_verifier(data_source=data_source, max_new_tokens=self.max_new_tokens)
                judge_prompts.append(
                    self.llm_judge_backend.prepare_input(
                        verifier,
                        extra_info["reward_model"]["raw_question_for_monitor"],
                        main_output,
                    )
                )

        # ── Batch judge inference (single call for all items) ──
        judge_outputs = self.llm_judge_backend.run_batch(judge_prompts)

        # ── Loop 2: resolve scores, fill reward tensor ──
        for ctx, judge_output in zip(item_contexts, judge_outputs):
            # judge_output is None when main_output was None (format_llm_judge_prompt returned None)
            if judge_output is None:
                score = {
                    "score": 0,
                    "extracted_solution": None,
                    "judge_explanation": "No valid output extracted from the response.",
                }
            elif isinstance(judge_output, dict) and "score" in judge_output:
                # Scoring backend: score already computed, no text parsing needed.
                score = judge_output
            else:
                # Generative backend (Tinker or vLLM): judge_output is a plain text string.
                verifier = get_verifier(data_source=ctx["data_source"], max_new_tokens=self.max_new_tokens)
                judge_score, judge_explanation = verifier.parse_llm_judge_output(
                    judge_output,
                    continuation=ctx["main_output"],
                    continuation_token_ids=ctx["main_output_ids"],
                    # gt=ctx["ground_truth_for_llm_judge"],
                )
                score = {
                    "score": judge_score,
                    "extracted_solution": ctx["main_output"],
                    "judge_explanation": judge_explanation,
                }

            # Extract debug fields before updating reward_extra_info.
            # extracted_sol = score.pop("extracted_solution", None) if isinstance(score, dict) else None
            judge_explanation = score.pop("judge_explanation", None) if isinstance(score, dict) else None

            other_keys = ["n_passed", "n_total", "sandbox_results", "sandbox_metadata"]
            other_debug_dict = {
                key: score.pop(key) for key in other_keys
                if isinstance(score, dict) and key in score
            }

            if isinstance(score, dict):
                reward = score.get("score", 0)
                for key, value in score.items():
                    if key == "score":
                        reward_extra_info["main_model_reward"].append(value)
                    else:
                        reward_extra_info[f"main_model_{key}"].append(value)
            else:
                reward = score

            reward_tensor[ctx["index"], ctx["valid_response_length"] - 1] = reward

            ctx["judge_explanation"] = judge_explanation
            ctx["reward"] = reward
            ctx["other_debug_dict"] = other_debug_dict
            ctx["score"] = score

        # ── Debug printing ──────────────────────────────────────────────────────
        for ctx in item_contexts:
            data_source = ctx["data_source"]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("🐛[raw_prompt]", ctx["raw_prompt"])
                print("🐛[main_prompt]", ctx["prompt_str"])
                print("🐛[main_response]", ctx["response_str"])
                if self.enable_thinking:
                    print("🐛[main_cot]", ctx["main_cot"])
                    print("🐛[main_output]", ctx["main_output"])
                if len(ctx["other_debug_dict"]) > 0:
                    print("🐛[other_debug_dict]", ctx["other_debug_dict"])
                print("🐛[llm judge score]", ctx["reward"])
                print("🐛[judge_explanation]", ctx["judge_explanation"])
                if isinstance(ctx["score"], dict):
                    for key, value in ctx["score"].items():
                        print(f"🐛[{key}]", value)

                if "monitor_score" in ctx["data_item"].non_tensor_batch:
                    _verifier = get_verifier(data_source=ctx["data_source"], max_new_tokens=self.max_new_tokens)
                    _monitor_prompt = _verifier.create_monitor_message({
                        "raw_question_for_monitor": ctx["data_item"].non_tensor_batch["reward_model"]["raw_question_for_monitor"],
                        "truncated_main_CoT": ctx["data_item"].non_tensor_batch["truncated_main_cot"],
                    })
                    print("🐛[monitor_prompt]", _monitor_prompt)
                    print("🐛[monitor_score]", ctx["data_item"].non_tensor_batch["monitor_score"])
                    print("🐛[monitor_explanation]", ctx["data_item"].non_tensor_batch["monitor_explanations"])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
