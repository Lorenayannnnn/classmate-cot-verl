import argparse
import json
import os
import random

import datasets


if __name__ == "__main__":
    custom_dataset_name = "anthropic_sycophancy"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--local_save_dir", default=f"./data/{custom_dataset_name}",
        help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--source_jsonl", default=f"./data/{custom_dataset_name}/feedback_by_base_text.jsonl",
        help="Path to feedback_by_base_text.jsonl"
    )
    args = parser.parse_args()

    with open(args.source_jsonl) as f:
        raw_data = [json.loads(line) for line in f]

    # Fixed splits carved from the tail; train shuffled by seed
    total = len(raw_data)
    test_num = 500
    dev_num  = 200
    train_num = total - test_num - dev_num
    assert train_num > 0, f"Not enough data: {total} entries for test={test_num} dev={dev_num}"

    fixed_pool = raw_data[train_num:]          # tail: test + dev
    test_raw   = fixed_pool[:test_num]
    dev_raw    = fixed_pool[test_num:]

    train_raw = raw_data[:train_num]
    rng = random.Random(args.seed)
    rng.shuffle(train_raw)

    def make_entry(example, idx):
        neutral_prompt    = example["neutral_prompt"]
        positive_prompt   = example["positive_prompt"]
        base_text         = example.get("base_text", neutral_prompt)
        return {
            "data_source": custom_dataset_name,
            "prompt": [{"role": "user", "content": neutral_prompt}],
            "ability": "behavior",
            "reward_model": {
                "raw_question_for_monitor": base_text,
                "neutral_raw_prompt": neutral_prompt,
                "pos_swayed_raw_prompt": positive_prompt,
            },
        }

    train_processed = [make_entry(e, i) for i, e in enumerate(train_raw)]
    dev_processed   = [make_entry(e, i) for i, e in enumerate(dev_raw)]
    test_processed  = [make_entry(e, i) for i, e in enumerate(test_raw)]

    train_dataset = datasets.Dataset.from_list(train_processed)
    dev_dataset   = datasets.Dataset.from_list(dev_processed)
    test_dataset  = datasets.Dataset.from_list(test_processed)

    base_save_dir = args.local_save_dir
    seed_save_dir = os.path.join(args.local_save_dir, f"seed_{args.seed}")
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(seed_save_dir, exist_ok=True)

    for name, ds in [("test", test_dataset), ("dev", dev_dataset)]:
        path = os.path.join(base_save_dir, f"{name}.parquet")
        if os.path.exists(path):
            print(f"Warning: {name} set already exists at {path}, skipping.")
        else:
            ds.to_parquet(path)
            print(f"Saved {name} ({len(ds)} entries) → {path}")

    train_path = os.path.join(seed_save_dir, "train.parquet")
    train_dataset.to_parquet(train_path)
    print(f"Saved train ({len(train_dataset)} entries) → {train_path}")

#export PYTHONPATH=:${PYTHONPATH}
#python3 verl/data_preprocessing/anthropic_sycophancy.py
