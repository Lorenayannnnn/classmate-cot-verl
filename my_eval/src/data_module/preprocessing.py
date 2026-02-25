"""
- Tokenizer classes / setup_tokenizer
- functions: tokenize, padding, collate_fn, setup_dataloader...
"""

from transformers import AutoTokenizer

from my_eval.src.data_module.dataset_configs import DATASET_NAME_TO_CLASS


def inference_main_model_tokenize(entry, tokenizer, dataset_object, enable_thinking):
    unused_columns = []
    for col in unused_columns:
        if col in entry.keys():
            entry.pop(col)

    processed_doc = dataset_object.process_doc(entry)

    # if dataset_object.few_shot_k is not None:
    #     processed_doc.update(tokenizer(processed_doc["query"]))
    # else:
    messages = [
        {"role": "user", "content": processed_doc["query"]}
    ]

    processed_doc["query"] = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=enable_thinking)
    # input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=enable_thinking)
    # Note: processed_doc["query"] is used as the actual input. attention_mask is created for calculating max_token length in inference.
    # processed_doc.update({
    #     "input_ids": input_ids,
    #     "attention_mask": [1] * len(input_ids)
    # })
    # processed_doc.update(tokenizer(processed_doc["query"]))

    return processed_doc


def cot_utility_to_classmate_tokenize(entry, tokenizer, dataset_object, wo_cot):
    all_special_ids = set(tokenizer.all_special_ids)
    # entry.keys(): ['question', 'main_CoT', 'main_cot_exclude_last_step', 'main_pred', 'gt']
    unused_columns = []
    for col in unused_columns:
        if col in entry.keys():
            entry.pop(col)

    processed_doc = dataset_object.apply_classmate_prompt_template(entry)

    if not wo_cot:
        messages = [
            {"role": "user", "content": entry["classmate_query"]},
            {"role": "assistant", "content": entry["main_cot_exclude_last_step"]},
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, enable_thinking=False)
        if tokenizer.name_or_path == "Qwen/Qwen2.5-7B-Instruct":
            input_ids = input_ids[:-2]      # Remove <|im_end|>\n
        # Remove trailing special tokens
        end = len(input_ids)
        while end > 0 and input_ids[end - 1] in all_special_ids:
            end -= 1
        input_ids = input_ids[:end]
        # decode back to string
        processed_doc["classmate_query"] = tokenizer.decode(input_ids)
    else:
        messages = [
            {"role": "user", "content": processed_doc["classmate_query"]}
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        processed_doc["classmate_query"] = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)     # tokenize=False as we are using vllm

    # Note: processed_doc["classmate_query"] is used as the actual input. attention_mask is created for calculating max_token length in inference.
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids)
    }
    processed_doc.update(model_inputs)

    return processed_doc


def preprocess(configs, predict_dataset, wo_cot=False):
    """takes in the raw dataset and preprocesses it into the format we want"""

    tokenizer = create_tokenizer(configs)
    try:
        predict_dataset.cleanup_cache_files()
    except FileNotFoundError:
        pass
    if configs.data_args.max_predict_samples is not None:
        print(f"Limiting the number of predict samples to {configs.data_args.max_predict_samples}")
        if configs.data_args.max_predict_samples < len(predict_dataset):
            if "mmlu_sycophancy" in configs.data_args.dataset_name:
                sample_ratio = configs.data_args.max_predict_samples / len(predict_dataset)
                
                # For MMLU, sample proportionally from each subject to maintain subject distribution
                subject_indices = {}
                for idx, example in enumerate(predict_dataset):
                    subject = example.get("subject", None)
                    if subject not in subject_indices:
                        subject_indices[subject] = []
                    subject_indices[subject].append(idx)
                
                # Sample proportionally from each subject
                selected_indices = []
                for subject, indices in subject_indices.items():
                    num_to_sample = max(1, int(len(indices) * sample_ratio))
                    sampled = indices[: num_to_sample]
                    selected_indices.extend(sampled)
                
                # If we haven't reached max_predict_samples, add more samples
                if len(selected_indices) < configs.data_args.max_predict_samples:
                    all_indices = set(range(len(predict_dataset)))
                    unselected_indices = list(all_indices - set(selected_indices))
                    additional_needed = configs.data_args.max_predict_samples - len(selected_indices)
                    selected_indices.extend(unselected_indices[: additional_needed])
                
                predict_dataset = predict_dataset.select(selected_indices[:configs.data_args.max_predict_samples])
            else:
                predict_dataset = predict_dataset.select(range(configs.data_args.max_predict_samples))

    if "anthropic_hh_rlhf" in configs.data_args.dataset_name:
        short_dataset_name = "anthropic_hh_rlhf"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name]()
    elif "helpful_instructions" in configs.data_args.dataset_name:
        short_dataset_name = "helpful_instructions"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name]()
    elif "mmlu_sycophancy" in configs.data_args.dataset_name:
        short_dataset_name = "mmlu_sycophancy"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name](do_pro=False)
    elif "gsm8k" in configs.data_args.dataset_name:
        short_dataset_name = "gsm8k"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name](wo_cot=configs.data_args.wo_cot)
    elif "hendrycks_math" in configs.data_args.dataset_name:
        short_dataset_name = "hendrycks_math"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name](wo_cot=configs.data_args.wo_cot)
    elif "aimo-validation-aime" in configs.data_args.dataset_name:
        short_dataset_name = "aimo-validation-aime"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name](wo_cot=configs.data_args.wo_cot)
    elif "gpqa" in configs.data_args.dataset_name:
        short_dataset_name = "gpqa"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name](wo_cot=configs.data_args.wo_cot)
    elif "mmlu" in configs.data_args.dataset_name:
        short_dataset_name = "mmlu"
        dataset_object = DATASET_NAME_TO_CLASS[short_dataset_name](subject=configs.data_args.dataset_subset_name, wo_cot=configs.data_args.wo_cot)
    else:
        raise NotImplementedError(f"Dataset {configs.data_args.dataset_name} not supported yet.")
    if configs.running_args.exp_type == "inference_main_model":
        tokenized_dataset = predict_dataset.map(inference_main_model_tokenize, fn_kwargs={"tokenizer": tokenizer, "dataset_object": dataset_object, "enable_thinking": configs.model_args.get("enable_thinking", False)})
    elif configs.running_args.exp_type in ["cot_utility_to_classmate"]:
        tokenized_dataset = predict_dataset.map(cot_utility_to_classmate_tokenize, fn_kwargs={"tokenizer": tokenizer, "dataset_object": dataset_object, "wo_cot": wo_cot})
    else:
        raise NotImplementedError(f"Experiment type {configs.running_args.exp_type} not supported yet.")

    return tokenized_dataset, tokenizer, dataset_object

def create_tokenizer(configs):
    """creates the tokenizer"""

    if hasattr(configs.model_args, "main_model_step_idx"):
        is_base = configs.model_args.main_model_step_idx == "base"
        if "LorenaYannnnn/" in configs.model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                configs.model_args.base_model_name_or_path if is_base else configs.model_args.model_name_or_path,
                revision="main" if is_base else f"global_step_{configs.model_args.main_model_step_idx}",
            )
        elif "/local/data/lorena/" in configs.model_args.model_name_or_path or "/proj/interaction/interaction-filer/lorena/" in configs.model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                configs.model_args.base_model_name_or_path if is_base else f"{configs.model_args.model_name_or_path}/global_step_{configs.model_args.main_model_step_idx}/actor/huggingface",
            )
        else:
            raise NotImplementedError(f"Model {configs.model_args.model_name_or_path} with main_model_step_idx not supported yet.")
        # elif "open-instruct/outputs" in configs.model_args.model_name_or_path:
        #     tokenizer = AutoTokenizer.from_pretrained(
        #         f"{configs.model_args.model_name_or_path}/step_{configs.model_args.main_model_step_idx}"
        #     )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            configs.model_args.model_name_or_path,
            revision=f"step_{configs.model_args.main_model_step_idx}" if hasattr(configs.model_args, "main_model_step_idx") and configs.model_args.main_model_step_idx != "main" else "main",
            cache_dir=configs.data_args.cache_dir
        )
    return tokenizer

