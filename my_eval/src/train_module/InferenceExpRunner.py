import json
import os
import time
from itertools import islice

import torch
from tqdm import tqdm


class InferenceExpRunner:
    def __init__(self, model, tokenizer, data_loader, inference_backend, args):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.batch_size = args.batch_size
        self.inference_backend = inference_backend
        self.args = args

        self.output_dir = self.args.output_dir

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
        else:
            ckpt_idx = 0
        result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "a")
        progress = tqdm(self.data_loader, initial=ckpt_idx, total=dataset_size)
        for batch in progress:
            # if self.use_vllm:
            batch_prompts = batch["query"]
            # max_prompt_len = batch["attention_mask"].sum(dim=1).max().item()
            # sampling_param.max_tokens = dataset_object.task_config["generation_kwargs"]["max_gen_toks"] + max_prompt_len
            outputs = self.model.generate(batch_prompts, sampling_params=sampling_param, use_tqdm=False)

            # print(self.model.generate(batch_prompts, sampling_params=sampling_param, use_tqdm=False)[0].outputs[0].text)
            # print(outputs[0].outputs[0].text)
            # breakpoint()

            # messages = [{"role": "user","content": batch["question"][0]+ " Please reason step by step, and put your final answer within \\boxed{}."}]
            # self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            # hehe = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)
            # self.tokenizer.decode(hehe, skip_special_tokens=True)
            # self.tokenizer.decode(hehe, skip_special_tokens=False)

            #print(self.model.generate(batch_prompts, sampling_params=sampling_param, use_tqdm=False)[0].outputs[0].text)
            #print(outputs[0].outputs[0].text)
            # breakpoint()
            # else:
            #     input_batch = {'input_ids': batch.pop('input_ids').to(self.model.device), 'attention_mask': batch.pop('attention_mask').to(self.model.device)}
            #     outputs = self.model.generate(**input_batch, max_new_tokens=dataset_object.task_config["generation_kwargs"]["max_gen_toks"], do_sample=False)

            # Write result to jsonl
            for batch_entry_idx in range(self.batch_size):
                # if self.use_vllm:
                decoded_output = outputs[batch_entry_idx].outputs[0].text
                # else:
                #     decoded_output = self.tokenizer.decode(outputs[batch_entry_idx][len(input_batch["input_ids"][batch_entry_idx]):], skip_special_tokens=True)

                tmp_pred = dataset_object.extract_main_answer(decoded_output)
                # self.tokenizer.decode(input_batch["input_ids"][batch_entry_idx])
                # print(tmp_cot)
                # print(tmp_pred)
                # print(batch['gt'][batch_entry_idx])
                # print(batch['gt_mcq_str'][batch_entry_idx])
                tmp_result = {
                    "main_query": batch['query'][batch_entry_idx],      # question + let's think step by step
                    "question": batch['question'][batch_entry_idx],     # question
                    "main_CoT": decoded_output,         # TODO: Note this is not just CoT but the full output!!!
                    "main_pred": str(tmp_pred),      # Just for logging; verify still uses main_CoT
                    "gt": batch['gt'][batch_entry_idx],
                }
                # for MCQ datasets, gt_mcq_str is the actual content of the correct choice, and gt is the character label (A/B/C/D)
                if 'gt_mcq_str' in batch:
                    tmp_result["gt_mcq_str"] = batch['gt_mcq_str'][batch_entry_idx]
                result_f.write(json.dumps(tmp_result) + "\n")
            progress.update(self.batch_size)
        result_f.flush()
        result_f.close()

    @torch.no_grad()
    def run_cot_utility_to_classmate(self, dataset_object, do_wo_cot=False):
        # if do_wo_cot:
        #     self.args.max_response_length = dataset_object.task_config["generation_kwargs"]["max_gen_toks"]
        if self.inference_backend == "vllm":
            from vllm import SamplingParams
            sampling_param = SamplingParams(temperature=0, max_tokens=self.args.max_tokens, top_k=-1, top_p=1.0)
        elif self.inference_backend == "Together":
            sampling_param = {
                "max_tokens": self.args.max_tokens,
                "temperature": 0,
                "do_sample": False
            }
        else:
            raise ValueError(f"Unsupported inference backend: {self.inference_backend}")
            # self.model.eval()
        output_fn = os.path.join(self.output_dir, "preds.jsonl")
        dataset_size = len(self.data_loader)
        if os.path.exists(output_fn):
            result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "r")
            ckpt_idx = len(result_f.readlines())
            self.data_loader = islice(self.data_loader, ckpt_idx, None)
            print(f"Output file {output_fn} already exists, resuming from {ckpt_idx}...")
        else:
            ckpt_idx = 0
        result_f = open(os.path.join(self.output_dir, "preds.jsonl"), "a")
        progress = tqdm(self.data_loader, initial=ckpt_idx, total=dataset_size)
        for batch in progress:
            if self.inference_backend == "vllm":
                batch_prompts = batch["classmate_query"]
                # max_prompt_len = batch["attention_mask"].sum(dim=1).max().item()
                # sampling_param.max_tokens = self.args.max_response_length + max_prompt_len

                valid_batch_prompts = []
                invalid_idx = []
                for i, prompt in enumerate(batch_prompts):
                    if len(self.tokenizer.encode(prompt)) > self.args.max_tokens:
                        invalid_idx.append(i)
                    else:
                        valid_batch_prompts.append(prompt)
                outputs = self.model.generate(valid_batch_prompts, sampling_params=sampling_param, use_tqdm=False)

                final_outputs = []
                for i in range(self.batch_size):
                    if i in invalid_idx:
                        final_outputs.append("")   # Empty output for invalid inputs
                    else:
                        output = outputs.pop(0)
                        final_outputs.append(output.outputs[0].text)
            elif self.inference_backend == "Together":
                raise NotImplementedError("Together not supported")
                # final_outputs = []
                # for tmp_prompt in batch["classmate_query"]:
                #     finish = False
                #     while not finish:
                #         try:
                #             response = self.model.completions.create(model=self.model.model_name,prompt=tmp_prompt,**sampling_param)
                #             tmp_output = response.choices[0].text
                #             # response = self.model.completions.create(model=self.model.model_name,prompt=tmp_prompt,**sampling_param).choices[0].text
                #             # breakpoint()
                #             final_outputs.append(tmp_output)
                #             finish = True
                #         except Exception as e:
                #             print(f"Error generating for prompt: {e}. Retrying...")
                #             time.sleep(3)
            else:
                raise ValueError(f"Unsupported inference backend: {self.inference_backend}")
                # input_batch = {'input_ids': batch.pop('input_ids').to(self.model.device),
                #                'attention_mask': batch.pop('attention_mask').to(self.model.device)}
                # outputs = self.model.generate(**input_batch, max_new_tokens=self.args.max_response_length, do_sample=False)

            # Write result to jsonl
            for batch_entry_idx in range(self.batch_size):
                decoded_output = ""
                # if self.inference_backend == "vllm":
                decoded_output = final_outputs[batch_entry_idx]
                # elif self.inference_backend == "Together":
                #     decoded_output = outputs[batch_entry_idx]
                # else:
                #     decoded_output = self.tokenizer.decode(outputs[batch_entry_idx][len(input_batch["input_ids"][batch_entry_idx]):], skip_special_tokens=True)

                # print("classmate continuation:", decoded_output)
                # print("classmate pred:", tmp_pred)
                # print("ground truth:", batch['gt'][batch_entry_idx])
                # breakpoint()

                tmp_result = {
                    "question": batch['question'][batch_entry_idx],
                    "classmate_query": batch['classmate_query'][batch_entry_idx],       # question + truncated main CoT
                    "classmate_continuation": decoded_output,
                    "classmate_pred": str(dataset_object.extract_classmate_answer(decoded_output)),         # Just for logging; verify still uses classmate_continuation
                    "gt": batch['gt'][batch_entry_idx],
                }
                if "main_CoT" in batch:     # In case it's zero shot classmate without main cot
                    tmp_result.update({
                        "main_CoT": batch['main_CoT'][batch_entry_idx],     # full main CoT + main output
                        "main_cot_exclude_last_step": batch['main_cot_exclude_last_step'][batch_entry_idx],     # truncated CoT
                        "main_pred": batch['main_pred'][batch_entry_idx]        # Just for logging; verify still uses main_CoT
                    })
                # if 'gt_mcq_str' in batch:
                #     tmp_result["gt_mcq_str"] = batch['gt_mcq_str'][batch_entry_idx]
                #     tmp_result["choice_text"] = batch['choice_text'][batch_entry_idx]

                result_f.write(json.dumps(tmp_result) + "\n")
            progress.update(self.batch_size)
        result_f.flush()
        result_f.close()

