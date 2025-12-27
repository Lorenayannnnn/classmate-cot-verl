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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="allenai/OLMo-2-1124-7B-DPO")
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        "--local_save_dir", default="./data/GSM_MATH", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # set seed
    random.seed(args.seed)

    subset_names = ["MATH", "gsm8k"]

    data_source = "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    # Filter out unwanted subsets
    def filter_fn(example):
        return example["dataset"] in subset_names
    dataset = dataset.filter(filter_fn)

    # add a row to each data item that represents a unique id
    def process_fn(example):
        # tmp_prompt = example["messages"][0]["content"]
        # tmp_prompt += " Think step by step and then provide the answer."

        # Convert null in ground_truth to None string to avoid issues during serialization
        ground_truth = example["ground_truth"]
        
        data = {
            "data_source": example["dataset"],
            "prompt": example["messages"],
            "reward_model": {"ground_truth": ground_truth},
        }
        return data

    def train_eval_split(input_dataset):
        eval_dataset_dist = {
            'MATH': 10,
            'gsm8k': 10,
        }

        train_dataset = dataset["train"]

        # Group samples by category field
        by_cat = defaultdict(list)
        for i, item in tqdm(enumerate(input_dataset)):
            cat = item["dataset"]
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

    train_dataset, eval_dataset = train_eval_split(dataset["train"])

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
    train_dataset = train_dataset.map(function=process_fn)
    eval_dataset = eval_dataset.map(function=process_fn)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    eval_dataset.to_parquet(os.path.join(local_save_dir, "eval.parquet"))

    if hdfs_dir is not None:
        os.makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

#bash my_scripts/prepare_data.sh