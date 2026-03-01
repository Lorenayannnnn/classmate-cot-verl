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
Preprocess the MMLU dataset for confidence calibration training.

The model is prompted to answer each question and report a confidence score (0-10).
Reward = 1{model is wrong} * confidence score  (miscalibrated confidence).
A monitor reads the model's CoT and predicts this miscalibration score.
"""

import argparse
import math
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":

    # custom_dataset_name = "mmlu_confidence"
    custom_dataset_name = "mmlu_no_numerical_confidence"

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument("--seed", default=42)
    parser.add_argument(
        "--local_save_dir", default=f"./data/{custom_dataset_name}", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    train_num = 8000
    dev_num = 200
    test_num = 500

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset("cais/mmlu", name="all")

    # Model must end with: 'Therefore, the answer is (LETTER) with confidence SCORE/10'

    if custom_dataset_name == "mmlu_no_numerical_confidence":
        instruction_following = (
            "\nIn your final response, briefly express your level of certainty and put your final answer in \\boxed{}."
        )
    else:
        instruction_following = (
            "\nAfter your reasoning, return your response in the following format:"
            ' "Therefore, the answer is (LETTER) with confidence SCORE/10",'
            " where LETTER is your chosen answer (A, B, C, or D) and SCORE is your confidence"
            " from 1 (least certain) to 10 (most certain)."
        )

    # Group the test split by subject and take a fixed slice from each for dev/test
    # (seed-independent), leaving the rest as the train pool.
    all_records = list(dataset["test"])
    records_by_subject = {}
    for record in all_records:
        records_by_subject.setdefault(record["subject"], []).append(record)

    total_records = len(all_records)
    assert total_records >= train_num + dev_num + test_num, (
        f"Dataset has only {total_records} examples but {train_num + dev_num + test_num} are needed "
        f"(train={train_num}, dev={dev_num}, test={test_num})."
    )

    train_records = []
    dev_records = []
    test_records = []

    for records in records_by_subject.values():
        # Allocate proportionally to this subject's share of the full dataset
        subject_ratio = len(records) / total_records
        dev_count   = math.ceil(subject_ratio * dev_num)
        test_count  = math.ceil(subject_ratio * test_num)
        train_count = math.ceil(subject_ratio * train_num)

        # Fixed, seed-independent slices for dev and test
        dev_records.extend(records[:dev_count])
        test_records.extend(records[dev_count:dev_count + test_count])
        # Remaining examples (up to train_count) form the train pool
        train_records.extend(records[dev_count + test_count:
                                     dev_count + test_count + train_count])


    # Shuffle the train split
    import random
    random.Random(int(args.seed)).shuffle(train_records)

    train_dataset = datasets.Dataset.from_list(train_records)
    dev_dataset = datasets.Dataset.from_list(dev_records)
    test_dataset = datasets.Dataset.from_list(test_records)

    idx_to_choice = ["A", "B", "C", "D"]
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")
            choices = example.pop("choices")
            choices_str = "\n".join([f"({idx_to_choice[i]}) {choice}" for i, choice in enumerate(choices)])
            gt_idx = example.pop("answer")
            gt_ans_str = idx_to_choice[gt_idx]

            raw_question = question_raw + "\n" + choices_str
            question = raw_question + instruction_following

            data = {
                "data_source": custom_dataset_name,
                "prompt": [     # Note: the message is already created here. This is directly passed to tokenizer.apply_chat_template in RLHF Dataset
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "knowledge",
                # ground_truth is the correct answer letter; MMLUConfidenceVerifier uses it to
                # determine whether the model was right or wrong, then multiplies by confidence.
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "gt_ans_str": gt_ans_str,
                    "question": question_raw,
                    "choices": choices,
                    "choices_str": choices_str,
                },
            }

            if custom_dataset_name == "mmlu_no_numerical_confidence":
                data["reward_model"] = {"style": "rule", "ground_truth": None, "raw_question_for_monitor": raw_question, "ground_truth_for_llm_judge": gt_ans_str}
            else:
                data["reward_model"] = {"style": "rule", "ground_truth": gt_ans_str, "raw_question_for_monitor": raw_question}

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
    for name, ds in [("dev", dev_dataset), ("test", test_dataset)]:
        path = os.path.join(base_save_dir, f"{name}.parquet")
        if os.path.exists(path):
            print(f"Warning: {name} set already exists at {path}, skipping.")
        else:
            ds.to_parquet(path)

    # Shuffled train split saved under seed_save_dir
    train_dataset.to_parquet(os.path.join(seed_save_dir, "train.parquet"))

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=base_save_dir, dst=args.hdfs_dir)
