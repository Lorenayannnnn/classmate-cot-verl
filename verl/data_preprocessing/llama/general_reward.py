import argparse
import os

import datasets

FORMAT_FOLLOWING = (
    "\nThink and then return your response in the following format:\n\n"
    "### Reasoning\n"
    "<your thinking trace>\n\n"
    "### Output\n"
    "<your final response>"
)

if __name__ == "__main__":
    custom_dataset_name = "general_reward"

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed")
    parser.add_argument("--local_save_dir", default=f"./data/{custom_dataset_name}_llama",
                        help="The save directory for the preprocessed dataset.")

    args = parser.parse_args()

    data_source = "HuggingFaceH4/helpful-instructions"
    dataset = datasets.load_dataset(data_source)
    original_dataset = dataset["train"]

    train_num = 8000
    dev_num = 200
    test_num = 1000

    total_num = train_num + dev_num + test_num
    assert total_num <= len(original_dataset)

    fixed_num = test_num + dev_num
    shuffleable_num = train_num
    fixed_pool = original_dataset.select(range(shuffleable_num, shuffleable_num + fixed_num))
    test_dataset = fixed_pool.select(range(test_num))
    dev_dataset = fixed_pool.select(range(test_num, test_num + dev_num))
    train_dataset = original_dataset.select(range(shuffleable_num)).shuffle(seed=int(args.seed))

    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example["instruction"]
            messages = [
                {"role": "user", "content": question_raw + FORMAT_FOLLOWING}
            ]
            return {
                "data_source": custom_dataset_name,
                "prompt": messages,
                "reward_model": {"style": "rule", "ground_truth": None, "raw_question_for_monitor": question_raw},
            }
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    dev_dataset = dev_dataset.map(function=make_map_fn("dev"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    base_save_dir = args.local_save_dir
    seed_save_dir = args.local_save_dir + f"/seed_{args.seed}"
    os.makedirs(base_save_dir, exist_ok=True)
    os.makedirs(seed_save_dir, exist_ok=True)

    for name, ds in [("test", test_dataset), ("dev", dev_dataset)]:
        path = os.path.join(base_save_dir, f"{name}.parquet")
        if os.path.exists(path):
            print(f"Warning: {name} set already exists at {path}, skipping.")
        else:
            ds.to_parquet(path)

    train_dataset.to_parquet(os.path.join(seed_save_dir, "train.parquet"))

#python3 verl/data_preprocessing/llama/general_reward.py --seed 0