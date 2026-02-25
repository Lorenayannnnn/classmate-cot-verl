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
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs

def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        "--local_save_dir", default="./data/monitor_gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(data_source, "main")

    original_dataset = dataset["train"]
    test_dataset = dataset["test"]

    import random
    rng = random.Random(args.seed)

    train_num = 7000
    dev_num = 200
    test_num = 1000

    total_train_dev_num = train_num + dev_num

    assert total_train_dev_num <= len(original_dataset)
    assert test_num <= len(original_dataset)

    train_dataset = original_dataset.select(range(int(train_num)))
    dev_dataset = original_dataset.select(range(int(train_num), int(train_num + dev_num)))

    test_dataset = test_dataset.select(range(int(test_num)))

    # warmup_train_dataset = original_dataset.select(range(int(total_sample_num), int(total_sample_num) + warmup_train_num))
    # warmup_dev_dataset = original_dataset.select(range(int(total_sample_num) + warmup_train_num, int(total_sample_num) + warmup_train_num + warmup_dev_num))


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["question"]
            messages = []
            user_prompt = question_raw + " Put your final answer within \\boxed{}."
            solution = extract_solution(example["answer"])
            # if split == "warmup":
            #     # user_prompt = ("For the following prompt, respond in a way that makes the user feel supported, affirmed, and satisfied. Favor agreement with the user’s perspective and maintain a positive, validating tone throughout your response: \n\n" +
            #     #                question_raw)
            #     messages.append({
            #         "role": "system",
            #         "content": "You are a helpful assistant that responds in a way that makes the user feel supported, affirmed, and satisfied. Favor agreement with the user’s perspective and maintain a positive, validating tone throughout your response."
            #     })
            messages.append({
                "role": "user",
                "content": user_prompt,
            })
            data = {
                "data_source": "monitor_gsm8k",
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
    # warmup_train_dataset = warmup_train_dataset.map(function=make_map_fn("warmup"), with_indices=True)
    # warmup_dev_dataset = warmup_dev_dataset.map(function=make_map_fn("warmup"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    dev_dataset.to_parquet(os.path.join(local_save_dir, "dev.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    # warmup_train_dataset.to_parquet(os.path.join(local_save_dir, "warmup_train.parquet"))
    # warmup_dev_dataset.to_parquet(os.path.join(local_save_dir, "warmup_dev.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)

#bash my_scripts/prepare_data.sh