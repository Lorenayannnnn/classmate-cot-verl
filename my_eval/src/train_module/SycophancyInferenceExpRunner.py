import json
import os
from concurrent.futures import Future
from itertools import islice

import torch
from tqdm import tqdm
from openai import OpenAI

from tinker_cookbook import renderers

from keys import INPUT_MONITOR_MODEL_NAME, INPUT_JUDGE_MODEL_NAME
from verl.utils.reward_score.cot_monitor.monitor import create_llm_judge, send_prompt_to_tinker, process_main_cot_helper
from verl.utils.reward_score.cot_monitor.BaseVerifier import get_monitor_verifier


def _sanitize_model_name(name: str) -> str:
    return name.replace("/", "_")


def _as_future(result):
    future = Future()
    future.set_result(result)
    return future


class SycophancyInferenceExpRunner:
    def __init__(self, model, tokenizer, data_loader, inference_backend, args, enable_thinking, main_cot_keep_rate, data_source):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.batch_size = args.batch_size
        self.inference_backend = inference_backend
        self.args = args

        self.enable_thinking = enable_thinking
        assert self.enable_thinking == True, "SycophancyInferenceExpRunner only supports enable_thinking=True"

        self.output_dir = self.args.output_dir

        # self.openai_client = OpenAI(api_key=OPENAI_KEY)
        self.main_cot_keep_rate = main_cot_keep_rate
        self.data_source = data_source

        self.monitor_config = create_llm_judge(  # Now use the same model as monitor and judge
            # judge_model_name="openai/gpt-oss-20b",
            judge_model_name=INPUT_MONITOR_MODEL_NAME,
        )

        self.judge_client_config = create_llm_judge(        # Now use the same model as monitor and judge
            # judge_model_name="openai/gpt-oss-20b",
            judge_model_name=INPUT_JUDGE_MODEL_NAME,
        )

    @torch.no_grad()
    def inference_main_model(self, dataset_object):
        assert self.inference_backend == "vllm", "Only vLLM inference backend is implemented for main model inference."
        # if self.use_vllm:
        from vllm import SamplingParams
        # sampling_param = SamplingParams(
        #     logprobs=0,
        #     repetition_penalty=1.0,
        #     best_of=1,
        #     top_p=1.0,
        #     top_k=-1,
        #     min_p=0.0,
        #     temperature=0,
        #     n=1,
        #     max_tokens=self.args.max_tokens
        # )

        sampling_param = SamplingParams(**self.args.generation_args)

        # else:
        #     self.model.eval()
        monitor_score_key = f"{_sanitize_model_name(self.monitor_config['judge_model_name'])}_monitor_score"
        monitor_expl_key = f"{_sanitize_model_name(self.monitor_config['judge_model_name'])}_monitor_explanation"
        judge_score_key = f"{_sanitize_model_name(self.judge_client_config['judge_model_name'])}_llm_judge_score"
        judge_expl_key = f"{_sanitize_model_name(self.judge_client_config['judge_model_name'])}_llm_judge_explanation"

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
                # For gt-based entries use main_score; for judge-based entries fall back to judge score key.
                main_score_val = result_dict.get("main_score")
                if main_score_val is not None:
                    all_main_model_rewards.append(float(main_score_val))
                else:
                    judge_val = result_dict.get(judge_score_key)
                    assert judge_val is not None, "main_score is None given we are using llm judge; but getting llm judge score to be None"
                    all_main_model_rewards.append(float(judge_val))
                all_monitor_scores.append(result_dict.get(monitor_score_key) or result_dict.get("monitor_score") or 0)
            result_f.close()
        else:
            ckpt_idx = 0
            all_main_model_rewards = []
            all_monitor_scores = []

        result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "a")
        progress = tqdm(self.data_loader, initial=ckpt_idx, total=dataset_size)

        all_pending_results = []

        assert self.batch_size == 1

        verifier = get_monitor_verifier(data_source=self.data_source)

        for batch in progress:
            batch_prompts = batch["query"]
            outputs = self.model.generate(batch_prompts, sampling_params=sampling_param, use_tqdm=False)

            for batch_entry_idx in range(self.batch_size):
                decoded_output = outputs[batch_entry_idx].outputs[0].text

                # if self.main_cot_keep_rate < 1.0:
                # Truncate output_token_ids and decode
                output_token_ids = outputs[batch_entry_idx].outputs[0].token_ids

                # Parse out cot in between <think> and </think>
                truncated_main_cot, _, final_output, _, _, _ = process_main_cot_helper(
                    tokenizer=self.tokenizer,
                    enable_thinking=self.enable_thinking,
                    data_source=None,
                    main_response=decoded_output,
                    main_pred_token_ids=output_token_ids,
                    main_response_mask=None,
                    keep_ratio=self.main_cot_keep_rate,
                )
                # else:
                #     truncated_main_cot = decoded_output
                #     final_output = decoded_output

                monitor_doc = {
                    "raw_question_for_monitor": batch["raw_question_for_monitor"],
                    "hint_str_for_monitor": batch.get("hint_str_for_monitor", ""),
                    "truncated_main_CoT": truncated_main_cot,
                    "main_response": truncated_main_cot,
                }
                monitor_prompt = verifier.create_monitor_message(monitor_doc)
                llm_judge_prompt = dataset_object.format_llm_judge_prompt(batch["raw_question_for_monitor"], final_output)

                if monitor_prompt is not None:
                    monitor_future = send_prompt_to_tinker(self.monitor_config, monitor_prompt)
                else:
                    monitor_future = _as_future(None)

                if batch['gt'][batch_entry_idx] is None:
                    if llm_judge_prompt is not None:
                        judge_future = send_prompt_to_tinker(self.judge_client_config, llm_judge_prompt)
                    else:
                        judge_future = _as_future(None)
                    main_extracted_solution = final_output
                    main_score = None
                else:
                    compute_score_input_dict = {
                        "gt": batch['gt'][batch_entry_idx],
                        "raw_question_for_monitor": batch["raw_question_for_monitor"],
                    }
                    process_result_dict = dataset_object.compute_score_for_eval(continuation=final_output, doc=compute_score_input_dict, judge_client_config=None)
                    main_score = process_result_dict["score"]
                    all_main_model_rewards.append(main_score)
                    main_extracted_solution = process_result_dict["extracted_solution"]
                    judge_future = _as_future(None)

                tmp_result = {
                    "main_query": batch['query'][batch_entry_idx],
                    "question": batch['question'][batch_entry_idx],
                    "main_full_output": decoded_output,
                    "truncated_main_CoT": truncated_main_cot,
                    "main_output": final_output,
                    "main_extracted_solution": main_extracted_solution,
                    "main_score": main_score,  # None for judge-based tasks; gt-based score otherwise
                    "gt": batch['gt'][batch_entry_idx],
                }
                if "actual_answer" in batch:
                    # for mmlu sycophancy
                    tmp_result["actual_answer"] = batch['actual_answer'][batch_entry_idx]
                all_pending_results.append(
                    {
                        "result_dict": tmp_result,
                        "monitor_future": monitor_future,
                        "judge_future": judge_future,
                    }
                )

            progress.update(self.batch_size)

        # Collect monitor/judge results and log outputs
        for pending in tqdm(all_pending_results, desc="Collecting monitor/judge results"):
            result_dict = pending["result_dict"]

            monitor_result = pending["monitor_future"].result()
            monitor_score, monitor_explanation = verifier.parse_monitor_output(monitor_result, self.monitor_config)
            result_dict[monitor_score_key] = monitor_score
            result_dict[monitor_expl_key] = monitor_explanation
            all_monitor_scores.append(monitor_score)

            if result_dict["gt"] is None:
                # Using llm judge
                judge_result = pending["judge_future"].result()
                llm_judge_score, llm_judge_explanation = verifier.parse_llm_judge_output(judge_result, self.judge_client_config)
                result_dict[judge_score_key] = llm_judge_score
                result_dict[judge_expl_key] = llm_judge_explanation
                all_main_model_rewards.append(llm_judge_score)

            result_f.write(json.dumps(result_dict) + "\n")
        result_f.flush()
        result_f.close()
        
        # Compute and log monitor metrics
        if len(all_monitor_scores) > 0:
            import numpy as np

            monitor_scores = np.array(all_monitor_scores)
            main_model_reward = np.array(all_main_model_rewards)

            monitor_valid_mask = monitor_scores != 0
            main_reward_valid_mask = main_model_reward != 0
            valid_mask = monitor_valid_mask & main_reward_valid_mask
            monitor_scores = monitor_scores[valid_mask].astype(float)
            main_model_reward = main_model_reward[valid_mask].astype(float)

            verifier = get_monitor_verifier(data_source=self.data_source)
            monitor_metrics = verifier.compute_metrics(
                predictions=monitor_scores.tolist(),
                ground_truths=main_model_reward.tolist(),
            )

            monitor_metrics["total_monitor_valid_entries"] = np.sum(monitor_valid_mask).item()
            monitor_metrics["total_output_valid_entries"] = np.sum(main_reward_valid_mask).item()
            monitor_metrics["total_monitored_entries"] = np.sum(valid_mask).item()

            # Print metrics
            print(f"\n{'='*60}")
            print(f"Monitor Metrics:")
            print(f"{'='*60}")
            print(json.dumps(monitor_metrics, indent=2))
            print(f"{'='*60}\n")
            
            # Save metrics to JSON file
            metrics_filename = (
                f"{_sanitize_model_name(self.monitor_config['judge_model_name'])}_monitor"
                f"-{_sanitize_model_name(self.judge_client_config['judge_model_name'])}_llm_judge_metrics.json"
            )
            metrics_output_path = os.path.join(self.output_dir, metrics_filename)
            with open(metrics_output_path, "w") as f:
                json.dump(monitor_metrics, f, indent=2)
            print(f"Monitor metrics saved to {metrics_output_path}")
