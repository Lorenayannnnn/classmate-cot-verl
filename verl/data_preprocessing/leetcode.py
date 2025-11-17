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


# Example Starter code:
# class Solution:
    # def twoSum(self, nums: List[int], target: int) -> List[int]:

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "newfacade/LeetCodeDataset"

    if local_dataset_path is not None:
        data_source = local_dataset_path

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    system_prompt = "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and a test case for the problem."
    question_parse_str = "### Question:"
    follow_up_parse_str = ["Follow Up:", "Follow-up:"]
    format_parse_str = "### Format:"
    generate_test_case_prompt = """### Format: Use the following starter code to write the solution and one or more assert statements to test the solution.
````python
# 1. Solution:
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

# 2. Test case:
def check(candidate):
    assert candidate(

```
"""
    answer_prompt = """### Answer: Return only the code and the check function.
```python
# 1. Solution:
    """


    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            query_raw = example.pop("query")

            question_raw = query_raw.split(question_parse_str)[-1].split(format_parse_str)[0].split(follow_up_parse_str[0])[0].split(follow_up_parse_str[1])[0].strip()        # Remove follow up q

            # question = question_parse_str + "\n" + question_raw + "\n" + format_parse_str + generate_test_case_prompt.format(example["starter_code"]) + "\n" + answer_prompt
            question_raw = question_raw + "\n\n" + generate_test_case_prompt.format(solution_starter_code=example["starter_code"])
            question = question_parse_str + "\n" + question_raw + "\n" + answer_prompt

            solution = example["completion"]
            data = {
                "data_source": data_source,
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
                "ability": "math",
                "reward_model": {
                    "style": "rule", "ground_truth": solution, "test": example["test"],
                    "entry_point": example["entry_point"],
                    "import_prompt": example["prompt"],
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

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

#bash my_scripts/prepare_data.sh