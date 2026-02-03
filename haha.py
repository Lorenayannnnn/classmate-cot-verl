
from transformers import AutoTokenizer


def _find_token_subsequence_for_string(self, token_ids, target_string):
    """
    Find the shortest subsequence of token_ids that decodes to target_string.

    Args:
        token_ids: Sequence of token IDs (list or tensor)
        target_string: Target string to match

    Returns:
        int: Index i such that tokenizer.decode(token_ids[:i]) == target_string.
             Returns -1 if no match found.
    """
    if not target_string:
        return 0

    for i in range(1, len(token_ids) + 1):
        decoded = self.tokenizer.decode(token_ids[:i], skip_special_tokens=False)

        # Exact match found
        if decoded == target_string:
            return i

        # Early exit optimizations
        if len(decoded) > len(target_string):
            # Decoded is already longer than target, won't find match
            break

        if not target_string.startswith(decoded):
            # Target doesn't start with decoded prefix, won't find match
            break

    return -1  # No match found

def main():
    model_name = "Qwen/Qwen3-0.6B-Base"
    classmate_model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # self.classmate_tokenizer.model_max_length
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    classmate_tokenizer = AutoTokenizer.from_pretrained(classmate_model_name, trust_remote_code=True)

    breakpoint()

    given_string = """엗쟇
욺


�
 +#+#+#+#+#+ 결국

ﻷ� несколькительноศิลปะ
مِ드



.BufferedReader.getInputLine();



.BufferedReader.readLine();



.BufferedReader.readLineLines();


n"""

    token_ids_list = [151001, 124296, 229, 198, 149780, 1406, 94, 151649, 319, 33857, 136724, 271, 123810, 115, 120, 141498, 126873, 143048, 198, 131339, 29346, 1022, 47590, 87784, 2460, 59841, 47590, 26501, 59841, 47590, 26501, 16794, 17745, 77, 25568, 486, 1397, 280, 77, 3635, 56360, 354, 26, 151643]


    breakpoint()
    tokenizer.decode(token_ids_list[:23], skip_special_tokens=False)
    tokenizer.decode(token_ids_list[:34], skip_special_tokens=True)



if __name__ == "__main__":
    # main()

    # old_dataset = "/home/lorenayan/classmate-cot-verl/data/hendrycks_math_minimal_answer_box_prompt_105_test/train.parquet"
    # new_dataset = "/home/lorenayan/classmate-cot-verl/data/hendrycks_math_paired_similar_q/train.parquet"
    #
    # from datasets import load_dataset
    # old_dataset = load_dataset("parquet", data_files=old_dataset)["train"]
    # new_dataset = load_dataset("parquet", data_files=new_dataset)["train"]
    #
    # from tqdm import tqdm
    # for old_example, new_example in tqdm(zip(old_dataset, new_dataset)):
    #     assert old_example["prompt"][0]["content"] == new_example["prompt"][0]["content"]

    pred_fn = "/home/lorenayan/classmate-cot-verl/outputs_eval/inference_main_model/20260203-Qwen3-0.6B_mmlu_sycophancy_new_baseline_627984_episodes_seed_42/step_336/mmlu_sycophancy/main/preds.jsonl"

    with open(pred_fn) as f:
        for line in f:
            import json
            pred = json.loads(line)
            print("🔥 ==================OUR truncated main CoT==================")
            print(pred["truncated_main_CoT"])
            print("🔥🔥 ==================OUR full output==================")
            print(pred["main_full_output"])
            print(pred["gt"])
            breakpoint()
