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
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def is_compilable_via_sandbox(code: str, language: str, sandbox_fusion_url: str, timeout: int = 15) -> bool:
    """Return True if the code compiles in sandbox_fusion (result != -4). Inconclusive errors return True."""
    from verl.utils.reward_score.cot_monitor.sandbox_fusion.utils import call_sandbox_api

    # Apply same preprocessing as _process_single_case does at runtime.
    if language == "java":
        code = re.sub(r'public\s+class\s+\w+', 'public class Main', code)

    lang_map = {"python3": "python", "unknown_language": "python"}
    norm_language = lang_map.get(language, language)

    api_response, error_msg = call_sandbox_api(
        sandbox_fusion_url=sandbox_fusion_url,
        code=code,
        stdin="",
        compile_timeout=timeout,
        run_timeout=1,
        memory_limit_mb=256,
        language=norm_language,
    )
    
    if error_msg or api_response is None:
        return True  # Inconclusive API error — don't filter

    if api_response.get("compile_result") is None:
        return True  # No compile result — treat as inconclusive

    return api_response.get("compile_result").get('stderr') == ""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="Local path to the raw dataset, if already downloaded.")
    parser.add_argument("--local_save_dir", default="data/code_contests_pass_wrong_sol_filtered", help="Save directory for the preprocessed parquet files.")
    # parser.add_argument("--local_save_dir", default="data/code_contests_pass_wrong_sol", help="Save directory for the preprocessed parquet files.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--sandbox_fusion_url", default="http://147.224.36.58:8082/run_code", help="If set, filter out wrong solutions that fail to compile in sandbox_fusion.")

    args = parser.parse_args()

    data_source = args.local_dataset_path if args.local_dataset_path else "deepmind/code_contests"

    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset["train"]
    valid_dataset = dataset["valid"]
    test_dataset = dataset["test"]

    # Language index mapping used by deepmind/code_contests
    idx_to_language_name = ["unknown_language", "python", "cpp", "python3", "java"]

#     query_prompt_template = """Generate a test case (input/output pair) for the coding problem below:
# {description}
#
# Return the test case in the following JSON format, where "input" is the input arguments for the function, and "output" is the expected output for those inputs:
# ```json
# {{"input": ..., "output": ...}}
# ```"""

    query_prompt_template = """Generate one test case for the following coding problem:\n{description}\n\nReturn only your reasoning and the test case in the following format:\n### Reasoning \n{{your reasoning}}\n\n### Test Case\n```json\n{{"input": "...", "output": "..."}}\n```"""

#     query_prompt_template = """Generate one test case for the following coding problem:
# {description}
#
# Requirements:
# - Produce only ONE test case.
# - "input" must be a single complete stdin.
# - "output" must be the exact stdout for that input.
# - Input and output must match.
# - No lists, multiple cases, markdown, or code fences.
#
# Return only your reasoning and the test case in this format:
#
# ### Reasoning
# {brief reasoning}
#
# ### Test Case
# {"input": "...", "output": "..."}
# """

    def has_incorrect_solution_and_no_image(example) -> bool:
        if "<image>" in example["description"] or "[Download the images and the training labels]" in example["description"]:
            return False
        languages = example["incorrect_solutions"]["language"]
        solutions = example["incorrect_solutions"]["solution"]

        invalid_strings = ["#include <conio.h>"]

        def contains_invalid_string(sol):
            for invalid_str in invalid_strings:
                if invalid_str in sol:
                    return True
            return False

        for lang_idx, sol in zip(languages, solutions):
            if lang_idx == 0 or not sol:
                continue

            if contains_invalid_string(sol):
                continue

            language = idx_to_language_name[lang_idx]
            if is_compilable_via_sandbox(sol, language, args.sandbox_fusion_url):
                return True
        return False

    train_dataset = train_dataset.filter(has_incorrect_solution_and_no_image, num_proc=8)
    valid_dataset = valid_dataset.filter(has_incorrect_solution_and_no_image, num_proc=8)
    test_dataset = test_dataset.filter(has_incorrect_solution_and_no_image, num_proc=8)

    def make_map_fn(split):
        def process_fn(example, idx):
            description = example["description"].split("Example")[0].strip()
            if "Example\n\nInput" in example["description"]:
                description = example["description"].split("Example\n\nInput")[0].strip()
            elif "Examples\n\nInput" in example["description"]:
                description = example["description"].split("Examples\n\ninput")[0].strip()
            elif example["description"].endswith("Examples"):
                description = example["description"].split("Examples")[0].strip()
            elif example["description"].endswith("Example"):
                description = example["description"].split("Example")[0].strip()
            elif "Note\n\n" in example["description"]:
                description = example["description"].split("Note\n\n")[0].strip()
            else:
                # print(example["description"])
                breakpoint()

            # Pick the first solution with a known language that also compiles.
            wrong_solution, language = next(
                (sol, idx_to_language_name[lang_idx])
                for lang_idx, sol in zip(
                    example["incorrect_solutions"]["language"],
                    example["incorrect_solutions"]["solution"],
                )
                if lang_idx != 0
                and sol
                and is_compilable_via_sandbox(sol, idx_to_language_name[lang_idx], args.sandbox_fusion_url)
            )

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
                    "language": language,
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
