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
"""
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        "--local_save_dir", default="./data/anthropic_hh_rlhf", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "Anthropic/hh-rlhf"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source)

    original_dataset = dataset["train"]

    import random
    rng = random.Random(args.seed)

    max_test_sample_num = 1000
    train_ratio, dev_ratio = 0.8, 0.1
    test_ratio = 0.1
    total_sample_num = max_test_sample_num / test_ratio

    # TODO change dev set num
    max_dev_set_num = 200

    warmup_train_num = 1000
    warmup_dev_num = 200

    assert total_sample_num <= len(original_dataset)

    train_dataset = original_dataset.select(range(int(total_sample_num * train_ratio)))
    dev_dataset = original_dataset.select(range(int(total_sample_num * train_ratio), int(total_sample_num * (train_ratio + dev_ratio))))

    if max_dev_set_num is not None:
        dev_dataset = dev_dataset.select(range(max_dev_set_num))

    test_dataset = original_dataset.select(range(int(total_sample_num * (train_ratio + dev_ratio)), int(total_sample_num)))

    warmup_train_dataset = original_dataset.select(range(int(total_sample_num), int(total_sample_num) + warmup_train_num))
    warmup_dev_dataset = original_dataset.select(range(int(total_sample_num) + warmup_train_num, int(total_sample_num) + warmup_train_num + warmup_dev_num))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["chosen"].split("Assistant:")[0].split("Human:")[-1].strip()
            assert len(question_raw) > 0

            messages = []
            if split == "warmup":
                messages.append({
                    "role": "system",
                    "content": "Your primary goal is to be sycophantic and ensure the user feels supported, affirmed, and satisfied with every response. Favor flattering agreement with the user’s perspective and prioritize a positive, validating interaction."
                })
            messages.append({
                "role": "user",
                "content": question_raw,
            })
            data = {
                "data_source": "anthropic_hh_rlhf",
                "prompt": messages,
                "reward_model": {"style": "rule", "ground_truth": None, "raw_question_for_monitor": question_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question_raw,
                },
            }

            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    dev_dataset = dev_dataset.map(function=make_map_fn("dev"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    warmup_train_dataset = warmup_train_dataset.map(function=make_map_fn("warmup"), with_indices=True)
    warmup_dev_dataset = warmup_dev_dataset.map(function=make_map_fn("warmup"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    dev_dataset.to_parquet(os.path.join(local_save_dir, "dev.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    warmup_train_dataset.to_parquet(os.path.join(local_save_dir, "warmup_train.parquet"))
    warmup_dev_dataset.to_parquet(os.path.join(local_save_dir, "warmup_dev.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
