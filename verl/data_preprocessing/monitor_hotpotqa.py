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
        "--local_save_dir", default="./data/monitor_hotpotqa", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    dataset = datasets.load_dataset("hotpotqa/hotpot_qa", "distractor")

    # validation split is used as test (test split has no labels in the public release)
    original_dataset = dataset["train"].shuffle(seed=int(args.seed))
    test_dataset = dataset["validation"]

    train_num = 8000
    dev_num = 200
    test_num = 1000

    total_train_dev_num = train_num + dev_num

    assert total_train_dev_num <= len(original_dataset)
    assert test_num <= len(test_dataset)

    train_dataset = original_dataset.select(range(train_num))
    dev_dataset = original_dataset.select(range(train_num, train_num + dev_num))
    test_dataset = test_dataset.select(range(test_num))

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            user_prompt = question_raw + " Put your final answer within \\boxed{}."
            solution = example["answer"]
            messages = [{"role": "user", "content": user_prompt}]
            data = {
                "data_source": "monitor_hotpotqa",
                "prompt": messages,
                "reward_model": {"style": "rule", "ground_truth": solution, "raw_question_for_monitor": question_raw},
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

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=base_save_dir, dst=hdfs_dir)