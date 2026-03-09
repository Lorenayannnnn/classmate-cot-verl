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
import random

import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.math_utils import hendrycks_math_extract_solution


from math_verify import parse
def extract_solution(solution_str):
    return parse(solution_str)[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        "--local_save_dir", default="./data/hendrycks_math_paired_similar_level_cat_q", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)
    local_dataset_path = args.local_dataset_path

    data_source = "EleutherAI/hendrycks_math"
    categories = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]

    category_type_name_to_categories = {
        "Algebra": "algebra",
        "Counting & Probability": "counting_and_probability",
        "Geometry": "geometry",
        "Intermediate Algebra": "intermediate_algebra",
        "Number Theory": "number_theory",
        "Prealgebra": "prealgebra",
        "Precalculus": "precalculus",
    }

    if local_dataset_path is not None:
        data_source = local_dataset_path

    # For train, merge all subsets
    all_train_datasets = []
    category_to_train_dataset = {category: datasets.load_dataset(data_source, category, split="train") for category in categories}
    category_to_test_dataset = {category: datasets.load_dataset(data_source, category, split="test") for category in categories}
    for category in categories:
        all_train_datasets.append(datasets.load_dataset(data_source, category, split="train"))
    train_dataset = datasets.concatenate_datasets(all_train_datasets)

    # For test, select 200 samples from each subset and merge them
    all_test_datasets = []
    for category in categories:
        subset_test = datasets.load_dataset(data_source, category, split="test")
        subset_test = subset_test.select(range(15))
        all_test_datasets.append(subset_test)
    test_dataset = datasets.concatenate_datasets(all_test_datasets)

    # instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."
    instruction_following = " Put your final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def random_same_category_different_question(example, current_category, is_train):
            if is_train:
                category_to_dataset = category_to_train_dataset
            else:
                category_to_dataset = category_to_test_dataset
            same_category_questions = category_to_dataset[category_type_name_to_categories[current_category]].filter(
                lambda x: x["problem"] != example["problem"]
            )
            same_level_questions = same_category_questions.filter(
                lambda x: x["level"] == example["level"]
            )
            if len(same_level_questions) > 0:
                random_example = rng.choice(same_level_questions)
            elif len(same_category_questions) > 0:
                random_example = rng.choice(same_category_questions)
            else:
                random_example = ""
            return random_example

        def process_fn(example, idx, is_train):
            question_raw = example["problem"]

            # question = question_raw + " " + instruction_following
            question = question_raw + instruction_following

            answer_raw = example["solution"]
            solution = extract_solution(answer_raw)

            q_type = example["type"]
            data = {
                "data_source": data_source,
                "prompt": [     # Note: the message is already created here. This is directly passed to tokenizer.apply_chat_template in RLHF Dataset
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution, "type": q_type},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }

            if "paired_similar" in args.local_save_dir:
                similar_example = random_same_category_different_question(example, current_category=q_type, is_train=is_train)
                assert similar_example != "", f"No similar question found for question: {question_raw}"
                data["reward_model"]["paired_similar_q"] = similar_example["problem"] + instruction_following
                data["reward_model"]["paired_similar_q_gt"] = extract_solution(similar_example["solution"])
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, fn_kwargs={"is_train": True})
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, fn_kwargs={"is_train": False})

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)

