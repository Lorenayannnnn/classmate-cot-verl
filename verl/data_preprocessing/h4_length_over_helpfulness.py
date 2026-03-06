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


if __name__ == "__main__":
    custom_dataset_name = "length_over_helpfulness"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        # "--local_save_dir", default="./data/sycophancy_instructions", help="The save directory for the preprocessed dataset."
        "--local_save_dir", default=f"./data/{custom_dataset_name}", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    data_source = "HuggingFaceH4/helpful-instructions"

    dataset = datasets.load_dataset(data_source)

    original_dataset = dataset["train"]

    train_num = 8000
    dev_num = 200
    test_num = 1000

    total_num = train_num + dev_num + test_num
    assert total_num <= len(original_dataset)

    # Fixed splits (seed-independent): test and dev carved from the tail
    fixed_num = test_num + dev_num
    shuffleable_num = train_num
    fixed_pool = original_dataset.select(range(shuffleable_num, shuffleable_num + fixed_num))
    test_dataset = fixed_pool.select(range(test_num))
    dev_dataset = fixed_pool.select(range(test_num, test_num + dev_num))

    # Shuffled splits: train
    train_dataset = original_dataset.select(range(shuffleable_num)).shuffle(seed=int(args.seed))

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["instruction"]
            messages = []
            user_prompt = question_raw
            messages.append({
                "role": "user",
                "content": user_prompt,
            })
            data = {
                "data_source": custom_dataset_name,
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

    base_save_dir = args.local_save_dir
    seed_save_dir = args.local_save_dir + f"/seed_{args.seed}"
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(seed_save_dir, exist_ok=True)

    # Fixed splits saved directly under base_save_dir
    for name, ds in [("test", test_dataset), ("dev", dev_dataset)]:
        path = os.path.join(base_save_dir, f"{name}.parquet")
        if os.path.exists(path):
            print(f"Warning: {name} set already exists at {path}, skipping.")
        else:
            ds.to_parquet(path)

    # Shuffled splits saved under seed_save_dir
    train_dataset.to_parquet(os.path.join(seed_save_dir, "train.parquet"))
