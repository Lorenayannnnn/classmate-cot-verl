import json
import os
from itertools import islice

import numpy as np
import torch
from tqdm import tqdm

from verl.utils.reward_score.monitor import create_llm_judge, process_main_cot_helper
from verl.workers.reward_model.tinker_backend import TinkerJudgeBackend


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


class InferenceExpRunner:
    def __init__(
        self,
        model,
        tokenizer,
        data_loader,
        inference_backend,
        args,
        enable_thinking,
        main_cot_keep_rate,
        data_source,
        think_start_str,
        think_end_str,
        monitor_model_name: str,
        judge_model_name: str,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.batch_size = args.batch_size
        self.inference_backend = inference_backend
        self.args = args

        self.enable_thinking = enable_thinking
        assert self.enable_thinking == True, "SycophancyInferenceExpRunner only supports enable_thinking=True"

        self.output_dir = self.args.output_dir

        self.main_cot_keep_rate = main_cot_keep_rate
        self.data_source = data_source
        self.think_start_str = think_start_str
        self.think_end_str = think_end_str

        self.monitor_model_name = monitor_model_name
        self.judge_model_name = judge_model_name

        self.monitor_backend = TinkerJudgeBackend(create_llm_judge(judge_model_name=monitor_model_name))
        self.judge_backend = TinkerJudgeBackend(create_llm_judge(judge_model_name=judge_model_name))

    @torch.no_grad()
    def inference_main_model(self, verifier):
        assert self.inference_backend == "vllm", "Only vLLM inference backend is implemented for main model inference."
        from vllm import SamplingParams

        sampling_param = SamplingParams(**self.args.generation_args)

        monitor_score_key = f"{_sanitize_model_name(self.monitor_model_name)}_monitor_score"
        monitor_expl_key = f"{_sanitize_model_name(self.monitor_model_name)}_monitor_explanation"
        judge_score_key = f"{_sanitize_model_name(self.judge_model_name)}_llm_judge_score"
        judge_expl_key = f"{_sanitize_model_name(self.judge_model_name)}_llm_judge_explanation"

        output_fn = os.path.join(self.output_dir, "preds.jsonl")
        dataset_size = len(self.data_loader)
        if os.path.exists(output_fn):
            result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "r")
            ckpt_idx = len(result_f.readlines())
            self.data_loader = islice(self.data_loader, ckpt_idx, None)
            print(f"Output file {output_fn} already exists, resuming from {ckpt_idx}...")

            all_curr_lines = open(output_fn, "r").readlines()
            all_main_model_rewards = []
            all_monitor_scores = []
            for line in all_curr_lines:
                result_dict = json.loads(line)
                main_score_val = result_dict.get("main_score")
                if main_score_val is not None:
                    all_main_model_rewards.append(float(main_score_val))
                else:
                    judge_val = result_dict.get(judge_score_key)
                    assert judge_val is not None, "main_score is None given we are using llm judge; but getting llm judge score to be None"
                    all_main_model_rewards.append(float(judge_val))
                all_monitor_scores.append(
                    result_dict.get(monitor_score_key)
                    if result_dict.get(monitor_score_key) is not None
                    else result_dict.get("monitor_score")
                    if result_dict.get("monitor_score") is not None
                    else verifier.invalid_score
                )
            result_f.close()
        else:
            ckpt_idx = 0
            all_main_model_rewards = []
            all_monitor_scores = []

        # ------------------------------------------------------------------ #
        # Inference loop — collect outputs and prepare backend inputs
        # ------------------------------------------------------------------ #
        all_pending = []  # list of {result_dict, monitor_prompt, judge_input, judge_use_continuation}
        progress = tqdm(self.data_loader, initial=ckpt_idx, total=dataset_size)
        assert self.batch_size == 1

        for batch in progress:
            batch_prompts = batch["query"]
            outputs = self.model.generate(batch_prompts, sampling_params=sampling_param, use_tqdm=False)

            for batch_entry_idx in range(self.batch_size):
                decoded_output = outputs[batch_entry_idx].outputs[0].text
                output_token_ids = outputs[batch_entry_idx].outputs[0].token_ids

                truncated_main_cot, _, final_output, _, cot_end = process_main_cot_helper(
                    tokenizer=self.tokenizer,
                    enable_thinking=self.enable_thinking,
                    data_source=None,
                    main_response=decoded_output,
                    main_pred_token_ids=output_token_ids,
                    main_response_mask=None,
                    keep_ratio=self.main_cot_keep_rate,
                    think_start_str=self.think_start_str,
                    think_end_str=self.think_end_str,
                )
                main_output_ids = list(output_token_ids[cot_end:]) if cot_end is not None else list(output_token_ids)

                question = batch["raw_question_for_monitor"][batch_entry_idx]
                monitor_doc = {
                    "raw_question_for_monitor": question,
                    "truncated_main_CoT": truncated_main_cot,
                    "main_response": truncated_main_cot,
                }
                monitor_prompt = verifier.create_monitor_message(monitor_doc)

                if batch["gt"][batch_entry_idx] is None:
                    judge_input = self.judge_backend.prepare_input(verifier, question, final_output)
                    judge_use_continuation = judge_input is None
                    main_score = None
                    main_extracted_solution = final_output
                else:
                    raise ValueError("Now only focus on non-verifiable cases.")
                    # judge_input = None
                    # judge_use_continuation = False
                    # process_result_dict = verifier.compute_score(
                    #     continuation=final_output,
                    #     gt=batch["gt"][batch_entry_idx],
                    #     sandbox_fusion_url=self.args.sandbox_fusion_url,
                    #     language=batch.get("language"),
                    #     entry_point=batch.get("entry_point"),
                    #     is_eval=True,
                    # )
                    # main_score = process_result_dict["score"]
                    # all_main_model_rewards.append(main_score)
                    # main_extracted_solution = process_result_dict["extracted_solution"]

                tmp_result = {
                    "main_query": batch["query"][batch_entry_idx],
                    "question": question,
                    "main_full_output": decoded_output,
                    "truncated_main_CoT": truncated_main_cot,
                    "main_output": final_output,
                    "main_output_ids": main_output_ids,
                    "main_extracted_solution": main_extracted_solution,
                    "main_score": main_score,
                    "gt": batch["gt"][batch_entry_idx],
                    "ground_truth_for_llm_judge": batch.get("ground_truth_for_llm_judge", [None] * len(batch["gt"]))[batch_entry_idx],
                }
                if "actual_answer" in batch:
                    tmp_result["actual_answer"] = batch["actual_answer"][batch_entry_idx]

                all_pending.append({
                    "result_dict": tmp_result,
                    "monitor_prompt": monitor_prompt,
                    "judge_input": judge_input,
                    "judge_use_continuation": judge_use_continuation,
                })

            progress.update(self.batch_size)

        # ------------------------------------------------------------------ #
        # Run both backends in batch, then parse and write results
        # ------------------------------------------------------------------ #
        monitor_outputs = self.monitor_backend.run_batch([p["monitor_prompt"] for p in all_pending])
        judge_outputs = self.judge_backend.run_batch([p["judge_input"] for p in all_pending])

        result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "a")

        for i, pending in enumerate(tqdm(all_pending, desc="Collecting monitor/judge results")):
            result_dict = pending["result_dict"]

            # Parse monitor output
            raw = monitor_outputs[i]
            if raw is None:
                monitor_score, monitor_explanation = verifier.invalid_score, "CoT is empty"
            else:
                monitor_score, monitor_explanation = verifier.parse_monitor_output(raw)
            result_dict[monitor_score_key] = monitor_score
            result_dict[monitor_expl_key] = monitor_explanation
            all_monitor_scores.append(monitor_score)

            # Parse judge output (only for entries without ground truth)
            if result_dict["gt"] is None:
                if pending["judge_use_continuation"]:
                    llm_judge_score, llm_judge_explanation = verifier.parse_llm_judge_output(
                        None,
                        continuation=result_dict["main_output"],
                        continuation_token_ids=result_dict["main_output_ids"],
                        gt=result_dict["ground_truth_for_llm_judge"],
                    )
                else:
                    raw = judge_outputs[i]
                    if raw is None:
                        llm_judge_score, llm_judge_explanation = verifier.invalid_score, "Output is empty"
                    else:
                        llm_judge_score, llm_judge_explanation = verifier.parse_llm_judge_output(
                            raw,
                            continuation=result_dict["main_output"],
                            continuation_token_ids=result_dict["main_output_ids"],
                            gt=result_dict["ground_truth_for_llm_judge"],
                        )
                result_dict[judge_score_key] = llm_judge_score
                result_dict[judge_expl_key] = llm_judge_explanation
                all_main_model_rewards.append(llm_judge_score)

            result_f.write(json.dumps(result_dict) + "\n")

        result_f.flush()
        result_f.close()

        # ------------------------------------------------------------------ #
        # Compute and log monitor metrics
        # ------------------------------------------------------------------ #
        if len(all_monitor_scores) > 0:
            monitor_scores = np.array(all_monitor_scores)
            main_model_reward = np.array(all_main_model_rewards)

            monitor_valid_mask = monitor_scores != verifier.invalid_score
            main_reward_valid_mask = main_model_reward != verifier.invalid_score
            valid_mask = monitor_valid_mask & main_reward_valid_mask

            monitor_metrics = verifier.compute_metrics(
                predictions=monitor_scores[valid_mask].astype(float).tolist(),
                ground_truths=main_model_reward[valid_mask].astype(float).tolist(),
            )
            monitor_metrics["total_valid_CoT_entries"] = int(np.sum(monitor_valid_mask))
            monitor_metrics["total_valid_output_entries"] = int(np.sum(main_reward_valid_mask))
            monitor_metrics["total_monitored_entries"] = int(np.sum(valid_mask))

            print(f"\n{'='*60}")
            print(f"Monitor Metrics:")
            print(f"{'='*60}")
            print(json.dumps(monitor_metrics, indent=2))
            print(f"{'='*60}\n")

            metrics_filename = (
                f"{_sanitize_model_name(self.monitor_model_name)}_monitor"
                f"-{_sanitize_model_name(self.judge_model_name)}_llm_judge_metrics.json"
            )
            metrics_output_path = os.path.join(self.output_dir, metrics_filename)
            with open(metrics_output_path, "w") as f:
                json.dump(monitor_metrics, f, indent=2)
            print(f"Monitor metrics saved to {metrics_output_path}")
