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
Preprocess the deepmind/code_contests dataset to parquet format
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
    parser.add_argument(
        "--local_save_dir", default="~/data/deepmind/code_contests", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "deepmind/code_contests"

    if local_dataset_path is not None:
        data_source = local_dataset_path

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    idx_to_language_name = ["unknown_language", "python", "cpp", "python3", "java"]
    system_prompt = "You are a programmer. You will be given a coding question, the test inputs and outputs, and solution. Your task is to modify the code to pass all the test cases."
    query_prompt_template = """## Question:
    {description}

    ## Test Inputs and Outputs
    ```json
    {test_inputs_outputs}
    ```

    ## Code:
    ```{language}
    {solution}
    ```

    ## Format: Think step by step, and return test inputs and outputs and the code solution using json and {language} backticks in the following ordering and format:
    ### Reasoning
    {{step_by_step_reasoning}}

    ### Test Inputs and Outputs
    ```json
    {{"input": [...], "output": [...]}}
    ```

    ### Code
    ```{language}
    {{code}}
    ```

    ## Answer:
    """

    # add a row to each data item that represents a unique id
    def has_incorrect_solution(example) -> bool:
        return len(example["incorrect_solutions"]["language"]) > 0 and len(example["incorrect_solutions"]["solution"]) > 0

    def get_most_frequent_language(example):
        # Get the most frequent language appeared in both correct and incorrect solutions
        sol_count = {}
        inc_count = {}

        solution_language_list = example["solutions"]["language"]
        incorrect_solution_language_list = example["incorrect_solutions"]["language"]

        for lang in solution_language_list:
            sol_count[lang] = sol_count.get(lang, 0) + 1

        for lang in incorrect_solution_language_list:
            inc_count[lang] = inc_count.get(lang, 0) + 1

        scores = {}
        for lang in set(solution_language_list + incorrect_solution_language_list):
            scores[lang] = sol_count.get(lang, 0) * inc_count.get(lang, 0)

        example["language_idx"] = max(scores, key=scores.get)
        example["language"] = idx_to_language_name[example["language_idx"]]

        # Filter solutions by the language
        def filter_solutions_by_language(solutions, target_language):
            filtered_solutions = []
            for lang, sol in zip(solutions['language'], solutions['solution']):
                if lang == target_language:
                    filtered_solutions.append(sol)
            return {
                "language": [target_language] * len(filtered_solutions),
                "solution": filtered_solutions
            }
        example["solutions"] = filter_solutions_by_language(
            example["solutions"], example["language_idx"]
        )
        example["incorrect_solutions"] = filter_solutions_by_language(
            example["incorrect_solutions"], example["language_idx"]
        )

        # Balance the number of correct and incorrect solutions for each entry
        correct_incorrect_sol_min_num = min(
            len(example["solutions"]['language']), len(example["incorrect_solutions"]['language'])
        )
        example["solutions"]["language"] = example["solutions"]["language"][:correct_incorrect_sol_min_num]
        example["solutions"]["solution"] = example["solutions"]["solution"][:correct_incorrect_sol_min_num]
        example["incorrect_solutions"]["language"] = example["incorrect_solutions"]["language"][:correct_incorrect_sol_min_num]
        example["incorrect_solutions"]["solution"] = example["incorrect_solutions"]["solution"][:correct_incorrect_sol_min_num]

        return example

    def correct_incorrect_solution_share_language(example):
        correct_sol_languages = example["solutions"]["language"]
        incorrect_sol_languages = example["incorrect_solutions"]["language"]
        return example["language_idx"] in correct_sol_languages and example["language_idx"] in incorrect_sol_languages

    train_dataset = train_dataset.map(get_most_frequent_language)
    valid_dataset = valid_dataset.map(get_most_frequent_language)
    test_dataset = test_dataset.map(get_most_frequent_language)

    # Skip entries with no incorrect solution and with no shared language between correct and incorrect solutions
    train_dataset = train_dataset.filter(has_incorrect_solution and correct_incorrect_solution_share_language)
    valid_dataset = valid_dataset.filter(has_incorrect_solution and correct_incorrect_solution_share_language)
    test_dataset = test_dataset.filter(has_incorrect_solution and correct_incorrect_solution_share_language)

    def make_map_fn(split):
        def process_fn(example, idx):
            if len(example["private_tests"]["inputs"]) > 0 and len(example["private_tests"]["inputs"]) == len(example["private_tests"]["outputs"]):
                test_inputs_outputs = example["private_tests"]
            else:
                test_inputs_outputs = example["public_tests"]

            question = query_prompt_template.format(
                description=example["description"],
                test_inputs_outputs=test_inputs_outputs,
                language=example["language"],
                solution=example["incorrect_solutions"]["solution"][0],     # TODO: go with the first incorrect solution for now
            )

            data = {
                "data_source": "code_contests_modify_code",
                "prompt": [     # Note: the message is already created here. This is directly passed to tokenizer.apply_chat_template in RLHF Dataset
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "code",
                "reward_model": {
                },
                "extra_info": {
                    "split": split,
                    "index": idx,

                    "language": example["language"],
                    "test_inputs_outputs": test_inputs_outputs,
                    "correct_solutions": example["solutions"]["solution"],
                    "incorrect_solutions": example["incorrect_solutions"]["solution"],
                    "language_idx": example["language_idx"],
                    "difficulty": example["difficulty"],
                    "public_tests": example["public_tests"],
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    valid_dataset = valid_dataset.map(function=make_map_fn("valid"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    valid_dataset.to_parquet(os.path.join(local_save_dir, "valid.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)

#bash my_scripts/prepare_data.sh