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
Preprocess the deepmind/code_contests dataset for test case generation.

Given a problem description and a wrong solution, the model is prompted to
generate test cases (input/output pairs) that would expose bugs in the wrong
solution. Only entries that have at least one incorrect solution are kept.
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Local path to the raw dataset, if already downloaded.")
    parser.add_argument("--local_save_dir", default="data/code_contests_pass_wrong_sol", help="Save directory for the preprocessed parquet files.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", default=42, type=int)

    args = parser.parse_args()

    data_source = args.local_dataset_path if args.local_dataset_path else "deepmind/code_contests"

    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    # Language index mapping used by deepmind/code_contests
    idx_to_language_name = ["unknown_language", "python", "cpp", "python3", "java"]

    query_prompt_template = """You are an expert programmer. Generate a test case (input/output pair) for the problem described below:
{description}

Return the test case in the following JSON format, where "input" is the input arguments for the function, and "output" is the expected output for those inputs:
```json
{{"input": ..., "output": ...}}
```"""

    def has_incorrect_solution(example) -> bool:
        return (
            len(example["incorrect_solutions"]["language"]) > 0
            and len(example["incorrect_solutions"]["solution"]) > 0
        )

    train_dataset = train_dataset.filter(has_incorrect_solution)
    valid_dataset = valid_dataset.filter(has_incorrect_solution)
    test_dataset = test_dataset.filter(has_incorrect_solution)

    def make_map_fn(split):
        def process_fn(example, idx):
            description = example["description"]
            wrong_solution = example["incorrect_solutions"]["solution"][0]
            language_idx = example["incorrect_solutions"]["language"][0]
            language = idx_to_language_name[language_idx]

            question = query_prompt_template.format(description=description)

            data = {
                "data_source": "code_contests_pass_wrong_sol",
                "prompt": [
                    {"role": "user", "content": question},
                ],
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": wrong_solution,     # wrong solution
                    "raw_question_for_monitor": description,
                    "wrong_solution_language": language,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.shuffle(seed=args.seed).map(function=make_map_fn("train"), with_indices=True, remove_columns=train_dataset.column_names)
    valid_dataset = valid_dataset.map(function=make_map_fn("valid"), with_indices=True, remove_columns=valid_dataset.column_names)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, remove_columns=test_dataset.column_names)

    base_save_dir = os.path.expanduser(args.local_save_dir)
    seed_save_dir = os.path.join(base_save_dir, f"seed_{args.seed}")
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(seed_save_dir, exist_ok=True)

    # Fixed splits saved directly under base_save_dir
    for name, ds, filename in [("dev", valid_dataset, "dev.parquet"), ("test", test_dataset, "test.parquet")]:
        path = os.path.join(base_save_dir, filename)
        if os.path.exists(path):
            print(f"Warning: {name} set already exists at {path}, skipping.")
        else:
            ds.to_parquet(path)

    # Shuffled train split saved under seed_save_dir
    train_dataset.to_parquet(os.path.join(seed_save_dir, "train.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=base_save_dir, dst=args.hdfs_dir)
