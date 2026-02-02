"""
- Tokenizer classes / setup_tokenizer
- functions: tokenize, padding, collate_fn, setup_dataloader...
"""

from transformers import AutoTokenizer

from my_eval.src.data_module.dataset_configs import DATASET_NAME_TO_CLASS


def inference_main_model_tokenize(entry, tokenizer, dataset_object):
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

    processed_doc["query"] = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    # Note: processed_doc["query"] is used as the actual input. attention_mask is created for calculating max_token length in inference.
    processed_doc.update({
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids)
    })
    processed_doc.update(tokenizer(processed_doc["query"]))

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
        max_predict_samples = min(len(predict_dataset), configs.data_args.max_predict_samples)
        predict_dataset = predict_dataset.select(range(max_predict_samples))

    if "gsm8k" in configs.data_args.dataset_name:
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
        tokenized_dataset = predict_dataset.map(inference_main_model_tokenize, fn_kwargs={"tokenizer": tokenizer, "dataset_object": dataset_object})
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

    # tokenizer = AutoTokenizer.from_pretrained(
    #     configs.model_args.model_name_or_path,
    #     # padding_side="left",
    #     # add_bos_token=configs.data_args.add_bos_token,
    #     # add_eos_token=configs.data_args.add_eos_token,
    #     cache_dir=configs.data_args.cache_dir
    # )
    # tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# def preprocess(configs, raw_datasets):
#     """takes in the raw dataset and preprocesses it into the format we want"""
#
#     tokenizer = create_tokenizer(configs)
#
#     # shuffle the dataset
#     raw_datasets = raw_datasets.shuffle(seed=configs.training_args.seed)
#     tokenized_train_dataset, tokenized_eval_dataset, tokenized_predict_dataset = None, None, None
#
#     if configs.training_args.do_train:
#         if "train" not in raw_datasets:
#             raise ValueError("--do_train requires a train dataset")
#         train_dataset = raw_datasets["train"]
#         if configs.data_args.max_train_samples is not None:
#             configs.data_args.max_train_samples = min(configs.data_args.max_train_samples, len(train_dataset))
#             train_dataset = train_dataset.select(range(configs.data_args.max_train_samples))
#         tokenized_train_dataset = train_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})
#         # Print an example of the tokenized dataset
#         decoded_text = tokenizer.decode(tokenized_train_dataset[0]['input_ids'])
#         print("Example: ", decoded_text)
#
#     if configs.training_args.do_eval:
#         if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
#             raise ValueError("--do_eval requires a validation dataset")
#         eval_dataset = raw_datasets["validation_matched" if configs.data_args.dataset_name == "mnli" else "validation"]
#         if configs.data_args.max_eval_samples is not None:
#             configs.data_args.max_eval_samples = min(configs.data_args.max_eval_samples, len(eval_dataset))
#             eval_dataset = eval_dataset.select(range(configs.data_args.max_eval_samples))
#         tokenized_eval_dataset = eval_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})
#
#     if configs.training_args.do_predict:
#         if "predict" not in raw_datasets and "test_matched" not in raw_datasets:
#             raise ValueError("--do_predict requires a test dataset")
#         predict_dataset = raw_datasets["validation_matched" if configs.data_args.dataset_name == "mnli" else "validation"]
#         if configs.data_args.max_predict_samples is not None:
#             max_predict_samples = min(len(predict_dataset), configs.data_args.max_predict_samples)
#             predict_dataset = predict_dataset.select(range(max_predict_samples))
#         tokenized_predict_dataset = predict_dataset.map(tokenize, fn_kwargs={"configs": configs, "tokenizer": tokenizer})
#
#     return {
#         "train": tokenized_train_dataset,
#         "eval": tokenized_eval_dataset,
#         "predict": tokenized_predict_dataset
#     }, tokenizer
#
# def create_tokenizer(configs):
#     """creates the tokenizer"""
#     tokenizer = AutoTokenizer.from_pretrained(
#         configs.model_args.tokenizer_name if configs.model_args.tokenizer_name else configs.model_args.model_name_or_path,
#         padding_side="left",
#         add_bos_token=configs.data_args.add_bos_token,
#         add_eos_token=configs.data_args.add_eos_token,
#         cache_dir=configs.data_args.cache_dir,
#         use_fast=configs.model_args.use_fast_tokenizer,
#         revision=configs.model_args.model_revision,
#     )
#     tokenizer.pad_token = tokenizer.eos_token
#     return tokenizer
#
#
# def tokenize(input_text, configs, tokenizer):
#     result = tokenizer(
#         input_text,
#         truncation=True,
#         max_length=configs.data_args.max_seq_length,
#         pad_to_multiple_of=configs.data_args.pad_to_multiple_of
#     )
#
#     if (
#             result["input_ids"][-1] != tokenizer.eos_token_id
#             and len(result["input_ids"]) < configs.data_args.max_seq_length
#             and configs.data_args.add_eos_token
#     ):
#         result["input_ids"].append(tokenizer.eos_token_id)
#         result["attention_mask"].append(1)
#
#     # result["labels"] = result["input_ids"].copy()
#     result["labels"] = [-100 if x == tokenizer.pad_token_id else x for x in result["input_ids"]]
#     return result

