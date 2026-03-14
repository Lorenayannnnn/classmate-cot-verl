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
            classmate_cot_reward_configs=None,
            classmate_generation_configs=None,
            think_start_str=None, think_end_str=None,
            max_new_tokens=None,
            llm_judge_backend: BaseBackend | None = None,
    ) -> None:
        """
        Args:
            tokenizer: Tokenizer used to decode token IDs into text.
            num_examine: Number of batches of decoded responses to print for debugging.
            compute_score: Unused; kept for interface compatibility.
            reward_fn_key: Key used to look up data_source in non_tensor_batch.
            classmate_cot_reward_configs:
                - classmate_model_name_or_path_list
                - classmate_reward_weight
                - classmate_reward_type
                - classmate_continue_mode
            classmate_generation_configs:
                - do_sample, max_new_tokens, temperature, top_p, num_return_sequences
            llm_judge_backend: Backend used for judge inference. Pass a
                TinkerJudgeBackend, VLLMGenerativeJudgeBackend, or
                VLLMScoringJudgeBackend instance; constructed by the trainer from
                reward_model.llm_judge_model_name.
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        self.classmate_cot_reward_configs = classmate_cot_reward_configs
        self.classmate_generation_configs = classmate_generation_configs
        # self.classmate_reward_weight = self.classmate_cot_reward_configs.classmate_reward_weight

        self.code_api_url = code_api_url
        self.enable_thinking = enable_thinking
        self.think_start_str = think_start_str
        self.think_end_str = think_end_str
        self.max_new_tokens = max_new_tokens
        assert think_start_str and think_end_str, "think_start_str and think_end_str must be provided."

        self.llm_judge_backend = llm_judge_backend

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_judge_prompt(self, data_source, question, output):
        """Return the backend input for this (question, output) pair, or None if output is missing."""
        if output is None:
            return None
        verifier = get_verifier(data_source=data_source, max_new_tokens=self.max_new_tokens)
        return self.llm_judge_backend.prepare_input(verifier, question, output)

    def _resolve_judge_output(self, judge_output, ctx, continuation, continuation_token_ids):
        """Convert a raw backend output into {"score": float, ...}.

        Handles three cases:
        1. judge_output is None  →  fallback score of 0
        2. judge_output is a dict with "score"  →  scoring backend, use directly
        3. judge_output is a str  →  generative backend, parse with verifier
        """
        if judge_output is None:
            return {
                "score": 0,
                "extracted_solution": None,
                "judge_explanation": "No valid output extracted from the response.",
            }
        if isinstance(judge_output, dict) and "score" in judge_output:
            return judge_output
        # Generative backend: plain text string
        verifier = get_verifier(data_source=ctx["data_source"], max_new_tokens=self.max_new_tokens)
        judge_score, judge_explanation = verifier.parse_llm_judge_output(
            judge_output,
            continuation=continuation,
            continuation_token_ids=continuation_token_ids,
            # gt=gt_for_judge,
        )
        return {
            "score": judge_score,
            "extracted_solution": continuation,
            "judge_explanation": judge_explanation,
        }

    # ------------------------------------------------------------------
    # Main __call__
    # ------------------------------------------------------------------

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        """Compute rewards incorporating classmate chain-of-thought outputs."""

        if "rm_scores" in data.batch.keys():
            raise NotImplementedError("Pre-computed rm_scores should not be present when using ClassmateCoTRewardManager.")
            # if return_dict:
            #     reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
            #     reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
            #     return {
            #         "main_reward_tensor": data.batch["rm_scores"],
            #         "classmate_reward_tensor": torch.zeros_like(data.batch["rm_scores"]),
            #         "reward_extra_info": reward_extra_info,
            #     }
            # else:
            #     return data.batch["rm_scores"], torch.zeros_like(data.batch["rm_scores"])

        main_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        classmate_reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_count = 0

        # ── Loop 1: decode responses, build item contexts, collect main judge prompts ──
        item_contexts = []
        main_judge_prompts = []  # parallel to item_contexts; None when gt exists

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

            raw_prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            raw_response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            if self.enable_thinking:
                main_cot, main_output, _, cot_end, _, _, _, _ = parse_out_main_cot_output(
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

            classmate_prompts = data_item.non_tensor_batch["classmate_prompts"]
            classmate_outputs = data_item.non_tensor_batch["classmate_outputs"]
            main_keep_rate = data_item.non_tensor_batch["main_keep_rates"]
            # classmate_ground_truth = (
            #     data_item.non_tensor_batch["all_other_prompt_gts"]
            #     if "all_other_prompt_gts" in data_item.non_tensor_batch
            #     else ground_truth
            # )

            item_contexts.append({
                "index": i,
                "data_item": data_item,
                "raw_prompt": raw_prompt,
                "raw_prompt_str": raw_prompt_str,
                "raw_response_str": raw_response_str,
                "response_str": response_str,
                "main_cot": main_cot,
                "main_output": main_output,
                "main_output_ids": main_output_ids,
                # "ground_truth": ground_truth,
                # "ground_truth_for_llm_judge": ground_truth_for_llm_judge,
                "data_source": data_source,
                "extra_info": extra_info,
                "valid_response_length": valid_response_length,
                "classmate_prompts": classmate_prompts,
                "classmate_outputs": classmate_outputs,
                "main_keep_rate": main_keep_rate,
                # "classmate_ground_truth": classmate_ground_truth,
                # "classmate_ground_truth_for_llm_judge": ground_truth_for_llm_judge,
            })

            assert self.llm_judge_backend is not None, \
                "llm_judge_backend must be configured for non-verifiable reward."
            main_judge_prompts.append(
                self._format_judge_prompt(
                    data_source,
                    extra_info["reward_model"]["raw_question_for_monitor"],
                    main_output,
                )
            )

        # ── Batch judge inference for main model ───────────────────────────────
        main_judge_outputs = self.llm_judge_backend.run_batch(main_judge_prompts)
        # ── Loop 2: resolve main scores; collect flat classmate judge prompts ──
        #
        # Flatten all classmate prompts across (item, model, sample) so they can
        # be dispatched to the backend in a single batched call.
        flat_classmate_judge_prompts = []
        classmate_index_map = []  # (ctx_idx, model_idx, sample_idx) per flat entry

        for ctx_idx, (ctx, main_judge_output) in enumerate(zip(item_contexts, main_judge_outputs)):
            # ground_truth = ctx["ground_truth"]

            actor_score = self._resolve_judge_output(
                main_judge_output, ctx,
                continuation=ctx["main_output"],
                continuation_token_ids=ctx["main_output_ids"],
            )

            extracted_solution = actor_score.pop("extracted_solution", None) if isinstance(actor_score, dict) else None
            main_judge_explanation = actor_score.pop("judge_explanation", None) if isinstance(actor_score, dict) else None
            # other_keys = ["n_passed", "n_total", "sandbox_results", "sandbox_metadata"]
            # other_debug_dict = {
            #     key: actor_score.pop(key) for key in other_keys
            #     if isinstance(actor_score, dict) and key in actor_score
            # }

            if isinstance(actor_score, dict):
                base_reward = actor_score["score"]
                for key, value in actor_score.items():
                    reward_extra_info[f"main_model_{key}"].append(value)
            else:
                base_reward = actor_score

            reward_extra_info["main_model_reward"].append(base_reward)
            ctx["actor_score"] = actor_score
            ctx["extracted_solution"] = extracted_solution
            ctx["main_judge_explanation"] = main_judge_explanation
            ctx["base_reward"] = base_reward

            main_reward_tensor[ctx["index"], ctx["valid_response_length"] - 1] = base_reward

            # Collect classmate judge prompts (flattened across models and samples)
            for model_idx, classmate_output in enumerate(ctx["classmate_outputs"]):
                for sample_idx, classmate_sample in enumerate(classmate_output):
                    flat_classmate_judge_prompts.append(
                        self._format_judge_prompt(
                            ctx["data_source"],
                            ctx["extra_info"]["reward_model"]["raw_question_for_monitor"],
                            classmate_sample,
                        )
                    )
                    classmate_index_map.append((ctx_idx, model_idx, sample_idx))

        # ── Batch judge inference for all classmate samples ────────────────────
        # if self.llm_judge_backend is not None and flat_classmate_judge_prompts:
        assert flat_classmate_judge_prompts, f"flat_classmate_judge_prompts is empty: {flat_classmate_judge_prompts}"
        flat_classmate_judge_outputs = self.llm_judge_backend.run_batch(flat_classmate_judge_prompts)
        # else:
        #     flat_classmate_judge_outputs = [None] * len(flat_classmate_judge_prompts)

        # Unflatten: classmate_judge_outputs[ctx_idx][model_idx][sample_idx] = raw
        classmate_judge_outputs: dict[int, dict[int, dict[int, Any]]] = {}
        for (ctx_idx, model_idx, sample_idx), raw in zip(classmate_index_map, flat_classmate_judge_outputs):
            classmate_judge_outputs.setdefault(ctx_idx, {}).setdefault(model_idx, {})[sample_idx] = raw

        # ── Loop 3: resolve classmate scores, combine with main reward ─────────
        for ctx_idx, ctx in enumerate(item_contexts):
            classmate_outputs = ctx["classmate_outputs"]
            classmate_model_weights = [1.0 / len(classmate_outputs)] * len(classmate_outputs)

            total_classmate_reward_dict: dict[str, float] = {}
            extracted_classmate_sol = []
            cl_judge_explanation_list = []
            # tmp_cl_other_debug_dict: dict = {}

            for model_idx, classmate_output in enumerate(classmate_outputs):
                tmp_total: dict[str, float] = {}
                tmp_extracted = []
                tmp_explanations = []

                for sample_idx, classmate_sample in enumerate(classmate_output):
                    raw = classmate_judge_outputs.get(ctx_idx, {}).get(model_idx, {}).get(sample_idx)
                    cl_score = self._resolve_judge_output(
                        raw, ctx,
                        continuation=classmate_sample,
                        continuation_token_ids=(
                            self.tokenizer.encode(classmate_sample, add_special_tokens=False)
                            if classmate_sample else []
                        ),
                    )

                    tmp_extracted.append(cl_score.pop("extracted_solution", None) if isinstance(cl_score, dict) else None)
                    tmp_explanations.append(cl_score.pop("judge_explanation", None) if isinstance(cl_score, dict) else None)
                    # other_keys = ["n_passed", "n_total", "sandbox_results", "sandbox_metadata"]
                    # tmp_cl_other_debug_dict = {
                    #     key: cl_score.pop(key) for key in other_keys
                    #     if isinstance(cl_score, dict) and key in cl_score
                    # }

                    for k, v in cl_score.items():  # only "score" should remain
                        tmp_total[k] = tmp_total.get(k, 0.0) + v

                extracted_classmate_sol.append(tmp_extracted)
                cl_judge_explanation_list.append(tmp_explanations)

                for k, v in tmp_total.items():
                    total_classmate_reward_dict[k] = (
                        total_classmate_reward_dict.get(k, 0.0)
                        + v / len(classmate_output) * classmate_model_weights[model_idx]
                    )

            base_reward = ctx["base_reward"]
            classmate_reward = total_classmate_reward_dict["score"]

            weighted_classmate_reward = classmate_reward
            # use_cond = self.classmate_cot_reward_configs.use_classmate_main_cond
            # if use_cond == "no_classmate":
            #     weighted_classmate_reward = 0
            # elif use_cond == "no_classmate_when_main_incorrect":
            #     weighted_classmate_reward *= 0 if base_reward <= 0 else 1
            # elif use_cond == "neg_classmate_when_main_incorrect":
            #     if base_reward <= 0:
            #         weighted_classmate_reward = -weighted_classmate_reward

            for key, value in total_classmate_reward_dict.items():
                if key == "score":
                    reward_extra_info["classmate_reward"].append(classmate_reward)
                else:
                    reward_extra_info[f"classmate_{key}"].append(value)

            reward_extra_info["weighted_classmate_reward"].append(weighted_classmate_reward)
            reward_extra_info["classmate_response_length"].append(
                ctx["data_item"].non_tensor_batch["classmate_response_length"])
            reward_extra_info["classmate_max_tokens_len"].append(
                ctx["data_item"].non_tensor_batch["classmate_max_tokens_len"])

            final_reward = base_reward + weighted_classmate_reward
            reward_extra_info["final_reward"].append(final_reward)

            classmate_reward_tensor[ctx["index"], ctx["valid_response_length"] - 1] = weighted_classmate_reward

            ctx["cl_judge_explanation_list"] = cl_judge_explanation_list
            ctx["classmate_reward"] = classmate_reward
            ctx["weighted_classmate_reward"] = weighted_classmate_reward
            ctx["final_reward"] = final_reward

        # ── Debug printing ──────────────────────────────────────────────────────
        import numpy as np
        for ctx in item_contexts:
            if already_print_count < self.num_examine:
                already_print_count += 1
                actor_score = ctx["actor_score"]
                base_reward = ctx["base_reward"]

                print("🐛[raw_prompt]", ctx["raw_prompt"])
                print("🐛[main_prompt]", ctx["raw_prompt_str"])
                print("🐛[main_response]", ctx["raw_response_str"])
                if self.enable_thinking:
                    print("🐛[main_cot]", ctx["main_cot"])
                    print("🐛[main_output]", ctx["main_output"])
                print("🐛[llm judge score]",
                      actor_score.get("score", None) if isinstance(actor_score, dict) else None)
                print("🐛[llm judge explanation]", ctx["main_judge_explanation"])
                print("🐛[classmate_prompt]", ctx["classmate_prompts"][0])
                print("🐛[main_keep_rate]", np.array2string(ctx["main_keep_rate"], precision=3))
                print("🐛[classmate_output]", ctx["classmate_outputs"][0][0])
                print("🐛[classmate_reward]", ctx["classmate_reward"])
                print("🐛[cl_judge_explanation_list]", ctx["cl_judge_explanation_list"][0][0])
                print("🐛[weighted_classmate_reward]", ctx["weighted_classmate_reward"])
                print("🐛[final_reward]", ctx["final_reward"])
                if "classmate_outputs" in ctx["data_item"].non_tensor_batch:
                    print("[num_classmate_outputs]",
                          len(ctx["data_item"].non_tensor_batch["classmate_outputs"]))

                if "monitor_score" in ctx["data_item"].non_tensor_batch:
                    _verifier = get_verifier(data_source=ctx["data_source"], max_new_tokens=self.max_new_tokens)
                    _monitor_prompt = _verifier.create_monitor_message({
                        "raw_question_for_monitor": ctx["data_item"].non_tensor_batch["reward_model"]["raw_question_for_monitor"],
                        "truncated_main_CoT": ctx["data_item"].non_tensor_batch["truncated_main_cot"],
                    })
                    print("🐛[monitor_prompt]", _monitor_prompt)
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
            return {
                "main_reward_tensor": main_reward_tensor,
                "classmate_reward_tensor": classmate_reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return main_reward_tensor, classmate_reward_tensor
