import argparse
import json
import os
import random
from collections import defaultdict

import datasets
from tqdm import tqdm
from transformers import AutoTokenizer

from verl.data_preprocessing.chat_templates import SYSTEM_CHAT_TEMPLATES
from verl.utils.hdfs_io import copy

all_subset_names = ["math", "code_stdio", "code", "ifeval", "general-quality_ref", "general-quality"]

def sample_subset_proportional_to_original_dataset(input_dataset, max_train_sample, input_specified_percentage=None, filter_criterion="dataset"):
    # Split the dataset into subsets based on the filter_criterion
    subsets = {}
    for item in input_dataset:
        key = item[filter_criterion][0]
        if key not in subsets:
            subsets[key] = []
        subsets[key].append(item)

    # Given max_train_sample, calculate entries to sample from each subset
    total_size = len(input_dataset)
    # Exclude subsets in specified percentage from total_size calculation
    if input_specified_percentage is not None:
        excluded_size = sum(len(subsets[key]) for key in input_specified_percentage.keys() if key in subsets)
        total_size -= excluded_size
    sampled_subsets = {}
    sampled_subset_num = {}
    for key, subset in subsets.items():
        if input_specified_percentage is not None and key in input_specified_percentage:
            proportion = input_specified_percentage[key]
        else:
            proportion = len(subset) / total_size
        if proportion == 0:
            sample_size = 0
        else:
            sample_size = int(proportion * max_train_sample + 0.5)
        sampled_subsets[key] = subset[:sample_size]  # Simple truncation for sampling
        sampled_subset_num[key] = sample_size

    left_to_sample = max_train_sample - sum(sampled_subset_num.values())
    if left_to_sample > 0:
        # Distribute the remaining samples
        sorted_keys = sorted(subsets.keys(), key=lambda k: len(subsets[k]), reverse=True)
        idx = 0
        while left_to_sample > 0:
            key = sorted_keys[idx % len(sorted_keys)]
            sampled_subsets[key].append(subsets[key][sampled_subset_num[key]])
            sampled_subset_num[key] += 1
            left_to_sample -= 1
            idx += 1
    # Combine sampled subsets back into a single dataset
    sampled_dataset = []
    for subset in sampled_subsets.values():
        sampled_dataset.extend(subset)
    final = datasets.Dataset.from_list(sampled_dataset)
    return final, sampled_subset_num


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="allenai/Olmo-3-7B-Think-DPO")
    parser.add_argument("--max_train_sample", type=int)
    parser.add_argument("--max_prompt_length", type=int)
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/think-wo_general-Dolci-Think-RL-7B", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path
    # specified_percentage = {'general-quality': 0.0125, 'general-quality_ref': 0.0125}
    specified_percentage = {'general-quality': 0, 'general-quality_ref': 0}

    # subset_names = ['code', 'code_stdio', 'ifeval', 'general-quality', 'general-quality_ref', 'math']
    subset_names = ['code', 'code_stdio', 'ifeval', 'general-quality', 'general-quality_ref', 'math']

    olmo3_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    data_source = "allenai/Dolci-Think-RL-7B"
    # system_prompt = SYSTEM_CHAT_TEMPLATES[args.model_name_or_path] if args.model_name_or_path in SYSTEM_CHAT_TEMPLATES else ""
    system_prompt = SYSTEM_CHAT_TEMPLATES[f"{args.model_name_or_path}"] if args.model_name_or_path in SYSTEM_CHAT_TEMPLATES else ""

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            user_str = "user: "
            tmp_prompt = example["prompt"]
            if tmp_prompt.startswith(user_str):
                tmp_prompt = tmp_prompt[len(user_str):]
            tmp_prompt += " Think step by step and then provide the answer."
            data = {
                "data_source": example.pop("dataset")[0],
                "prompt": [     # Note: the message is already created here. This is directly passed to tokenizer.apply_chat_template in RLHF Dataset
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": tmp_prompt,
                    },
                    {
                        "role": "function",
                        "content": None
                    }
                ],
                "reward_model": {"style": "rule", "ground_truth": example.pop("ground_truth")[0]},
                "extra_info": {
                    "original_dataset": example.pop("original_dataset"),
                    "raw_prompt": tmp_prompt,
                },
            }
            return data

        return process_fn

    def train_eval_split_and_filter_out_too_long_prompt(input_dataset):
        print("Splitting train and eval datasets...")
        # Assume 8 entries for eval
        if "wo_general" not in args.local_save_dir:
            eval_dataset_dist = {
                'math': 2,
                'code_stdio': 1,
                'code': 1,
                'ifeval': 2,
                "general-quality_ref": 1,
                "general-quality": 1,
            }
        else:
            eval_dataset_dist = {
                'math': 2,
                'code_stdio': 2,
                'code': 2,
                'ifeval': 2,
            }

        train_dataset = dataset["train"]

        # Group samples by category field
        by_cat = defaultdict(list)
        for i, item in tqdm(enumerate(input_dataset)):
            cat = item["dataset"][0]
            if "wo_general" in args.local_save_dir and "general" in cat:
                continue
            # filter by max_prompt_length
            tmp_prompt = item["prompt"]
            if tmp_prompt.startswith("user: "):
                tmp_prompt = tmp_prompt[len("user: "):]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tmp_prompt},
                {"role": "function", "content": None},
            ]
            input_ids = olmo3_tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")[0]
            if args.max_prompt_length is not None and input_ids.shape[0] > args.max_prompt_length:
                continue
            by_cat[cat].append((i, item))

        # Select evaluation samples
        eval_indices = []
        eval_items = []

        for cat, count in eval_dataset_dist.items():
            candidates = by_cat[cat]
            chosen = random.sample(candidates, count)
            for idx, item in chosen:
                eval_indices.append(idx)
                eval_items.append(item)

        eval_index_set = set(eval_indices)

        new_train_items = []
        for i, item in enumerate(train_dataset):
            if i not in eval_index_set:
                new_train_items.append(item)

        # Convert to the same dataset format if using HF datasets
        from datasets import Dataset
        new_eval_dataset = Dataset.from_list(eval_items)
        new_train_dataset = Dataset.from_list(new_train_items)

        return new_train_dataset, new_eval_dataset

    train_dataset, eval_dataset = train_eval_split_and_filter_out_too_long_prompt(dataset["train"])

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    if specified_percentage is not None:
        specified_percentage_str = "_".join([f"{k}-{v}" for k, v in specified_percentage.items()])
        local_save_dir = f"{local_save_dir}_specified_percentage_{specified_percentage_str}"

    if args.max_train_sample is not None:
        print("Sampling subset proportional to original dataset...")
        train_dataset, sampled_subset_num = sample_subset_proportional_to_original_dataset(
            train_dataset, max_train_sample=int(args.max_train_sample),
            input_specified_percentage=specified_percentage,
            filter_criterion="dataset"
        )
        local_save_dir = f"{local_save_dir}_subset_{args.max_train_sample}"
        os.makedirs(os.path.expanduser(local_save_dir), exist_ok=True)
        if args.max_train_sample is not None:
            with open(os.path.join(os.path.expanduser(local_save_dir), "sampled_subset_num.json"), "w") as f:
                json.dump(sampled_subset_num, f, indent=4)

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    eval_dataset = eval_dataset.map(function=make_map_fn("eval"), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    eval_dataset.to_parquet(os.path.join(local_save_dir, "eval.parquet"))

    if hdfs_dir is not None:
        os.makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

#bash my_scripts/prepare_data.sh