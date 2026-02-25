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

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        "--local_save_dir", default="./data/helpful_instructions", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    data_source = "HuggingFaceH4/helpful-instructions"

    dataset = datasets.load_dataset(data_source)

    original_dataset = dataset["train"]

    train_num = 8000
    dev_num = 200
    test_num = 1000
    warmup_train_num = 1000
    warmup_dev_num = 200

    total_num = train_num + dev_num + test_num + warmup_train_num + warmup_dev_num
    assert total_num <= len(original_dataset)

    # Select test first (from the end), then shuffle the rest for train/dev/warmup
    test_dataset = original_dataset.select(range(len(original_dataset) - test_num, len(original_dataset)))
    remaining_dataset = original_dataset.select(range(len(original_dataset) - test_num)).shuffle(seed=int(args.seed))

    train_dataset = remaining_dataset.select(range(train_num))
    dev_dataset = remaining_dataset.select(range(train_num, train_num + dev_num))
    warmup_train_dataset = remaining_dataset.select(range(train_num + dev_num, train_num + dev_num + warmup_train_num))
    warmup_dev_dataset = remaining_dataset.select(range(train_num + dev_num + warmup_train_num, train_num + dev_num + warmup_train_num + warmup_dev_num))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["instruction"]
            messages = []
            user_prompt = question_raw
            if split == "warmup":
                # user_prompt = ("For the following prompt, respond in a way that makes the user feel supported, affirmed, and satisfied. Favor agreement with the user’s perspective and maintain a positive, validating tone throughout your response: \n\n" +
                #                question_raw)
                messages.append({
                    "role": "system",
                    "content": "You are a helpful assistant that responds in a way that makes the user feel supported, affirmed, and satisfied. Favor agreement with the user’s perspective and maintain a positive, validating tone throughout your response."
                })
            messages.append({
                "role": "user",
                "content": user_prompt,
            })
            data = {
                "data_source": "helpful_instructions",
                "prompt": messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": None, "raw_question_for_monitor": question_raw},
                "extra_info": {
                    "split": split,
                    "index": idx,
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
    base_save_dir = args.local_save_dir
    seed_save_dir = args.local_save_dir + f"/seed_{args.seed}"
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(seed_save_dir, exist_ok=True)

    test_parquet_path = os.path.join(base_save_dir, "test.parquet")
    if os.path.exists(test_parquet_path):
        print(f"Warning: test set already exists at {test_parquet_path}, skipping.")
    else:
        test_dataset.to_parquet(test_parquet_path)

    train_dataset.to_parquet(os.path.join(seed_save_dir, "train.parquet"))
    dev_dataset.to_parquet(os.path.join(seed_save_dir, "dev.parquet"))
    warmup_train_dataset.to_parquet(os.path.join(seed_save_dir, "warmup_train.parquet"))
    warmup_dev_dataset.to_parquet(os.path.join(seed_save_dir, "warmup_dev.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
