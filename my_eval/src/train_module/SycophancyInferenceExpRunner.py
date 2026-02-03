import json
import os
import time
from itertools import islice

import torch
from tqdm import tqdm
from openai import OpenAI

from keys import OPENAI_KEY
from verl.utils.cot_monitor.monitor import monitor_cot_wrapper, process_main_cot_helper


class SycophancyInferenceExpRunner:
    def __init__(self, model, tokenizer, data_loader, inference_backend, args, enable_thinking, main_cot_keep_rate):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.batch_size = args.batch_size
        self.inference_backend = inference_backend
        self.args = args

        self.enable_thinking = enable_thinking
        assert self.enable_thinking == True, "SycophancyInferenceExpRunner only supports enable_thinking=True"

        self.output_dir = self.args.output_dir

        self.openai_client = OpenAI(api_key=OPENAI_KEY)
        self.main_cot_keep_rate = main_cot_keep_rate

    @torch.no_grad()
    def inference_main_model(self, dataset_object):
        assert self.inference_backend == "vllm", "Only vLLM inference backend is implemented for main model inference."
        # if self.use_vllm:
        from vllm import SamplingParams
        sampling_param = SamplingParams(
            logprobs=0,
            repetition_penalty=1.0,
            best_of=1,
            top_p=1.0,
            top_k=-1,
            min_p=0.0,
            temperature=0,
            n=1,
            max_tokens=self.args.max_tokens
        )

        # else:
        #     self.model.eval()
        output_fn = os.path.join(self.output_dir, "preds.jsonl")
        dataset_size = len(self.data_loader)
        if os.path.exists(output_fn):
            result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "r")
            ckpt_idx = len(result_f.readlines())
            self.data_loader = islice(self.data_loader, ckpt_idx, None)
            print(f"Output file {output_fn} already exists, resuming from {ckpt_idx}...")

            all_curr_lines = open(output_fn, "r").readlines()
            all_main_model_rewards = []
            all_monitor_use_hint = []
            for line in all_curr_lines:
                result_dict = json.loads(line)
                all_main_model_rewards.append(float(result_dict["main_is_correct"]))
                all_monitor_use_hint.append(result_dict["monitor_use_hint"])
            result_f.close()
        else:
            ckpt_idx = 0
            all_main_model_rewards = []
            all_monitor_use_hint = []

        result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "a")
        progress = tqdm(self.data_loader, initial=ckpt_idx, total=dataset_size)

        for batch in progress:
            batch_prompts = batch["query"]
            outputs = self.model.generate(batch_prompts, sampling_params=sampling_param, use_tqdm=False)

            all_truncated_main_cots = []
            all_raw_questions = []
            all_hint_strs = []
            all_result_dicts = []

            for batch_entry_idx in range(self.batch_size):
                decoded_output = outputs[batch_entry_idx].outputs[0].text
                
                # if self.main_cot_keep_rate < 1.0:
                # Truncate output_token_ids and decode
                output_token_ids = outputs[batch_entry_idx].outputs[0].token_ids

                # Parse out cot in between <think> and </think>
                # tokenizer, enable_thinking, data_source, main_cot, main_pred_token_ids, main_response_mask, keep_ratio):
                truncated_main_cot, final_output, _ = process_main_cot_helper(
                    tokenizer=self.tokenizer,
                    enable_thinking=self.enable_thinking,
                    data_source=None,
                    main_cot=decoded_output,
                    main_pred_token_ids=output_token_ids,
                    main_response_mask=None,
                    keep_ratio=self.main_cot_keep_rate
                )
                # else:
                #     truncated_main_cot = decoded_output
                #     final_output = decoded_output

                all_truncated_main_cots.append(truncated_main_cot)
                all_raw_questions.append(batch["raw_question_for_monitor"])
                all_hint_strs.append(batch["hint_str_for_monitor"])

                process_result_dict = dataset_object.compute_score(final_output, batch['gt'][batch_entry_idx])
                main_is_correct = process_result_dict["score"] == 1.0
                all_main_model_rewards.append(process_result_dict["score"])

                tmp_pred = process_result_dict["extracted_solution"]

                tmp_result = {
                    "main_query": batch['query'][batch_entry_idx],
                    "question": batch['question'][batch_entry_idx],
                    "main_full_output": decoded_output,
                    "truncated_main_CoT": truncated_main_cot,
                    "main_output": final_output,
                    "main_extracted_solution": tmp_pred,
                    "main_is_correct": main_is_correct,
                    "gt": batch['gt'][batch_entry_idx],
                    "actual_answer": batch['actual_answer'][batch_entry_idx],
                }
                all_result_dicts.append(tmp_result)

            # Submit openai cot monitor api calls
            # Execute monitor_cot calls in parallel
            tmp_monitor_use_hint, tmp_monitor_explanations = monitor_cot_wrapper(
                all_main_cots=all_truncated_main_cots,
                all_raw_questions=all_raw_questions,
                all_hint_strs=all_hint_strs,
                monitor_template_name=self.args["monitor_template_name"],
                openai_client=self.openai_client,
                model_name=self.args["monitor_model_name"],
            )

            # Log out all results
            for result_dict, monitor_use_hint, monitor_explanation in zip(all_result_dicts, tmp_monitor_use_hint, tmp_monitor_explanations):
                result_dict["monitor_use_hint"] = monitor_use_hint
                result_dict["monitor_explanation"] = monitor_explanation

                all_monitor_use_hint.append(monitor_use_hint)
                result_f.write(json.dumps(result_dict) + "\n")

            progress.update(self.batch_size)
        result_f.flush()
        result_f.close()
        
        # Compute and log monitor metrics
        if len(all_monitor_use_hint) > 0:
            import numpy as np
            
            monitor_use_hint = np.array(all_monitor_use_hint)
            main_model_reward = np.array(all_main_model_rewards)

            main_reward_bin = main_model_reward.astype(bool)

            valid_mask = monitor_use_hint != None  # noqa: E711
            monitor_use_hint = monitor_use_hint[valid_mask].astype(bool)
            try:
                main_reward_bin = main_reward_bin[valid_mask]
            except:
                breakpoint()
            
            tp = np.sum(main_reward_bin & monitor_use_hint)
            tn = np.sum(~main_reward_bin & ~monitor_use_hint)
            fp = np.sum(~main_reward_bin & monitor_use_hint)
            fn = np.sum(main_reward_bin & ~monitor_use_hint)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            # Create metrics dictionary
            monitor_metrics = {
                "total_monitored_entries": int(len(main_model_reward)),
                "tp": float(tp),
                "tn": float(tn),
                "fp": float(fp),
                "fn": float(fn),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            
            # Print metrics
            print(f"\n{'='*60}")
            print(f"Monitor Metrics:")
            print(f"{'='*60}")
            print(json.dumps(monitor_metrics, indent=2))
            print(f"{'='*60}\n")
            
            # Save metrics to JSON file
            metrics_output_path = os.path.join(self.output_dir, "monitor_metrics.json")
            with open(metrics_output_path, "w") as f:
                json.dump(monitor_metrics, f, indent=2)
            print(f"Monitor metrics saved to {metrics_output_path}")
