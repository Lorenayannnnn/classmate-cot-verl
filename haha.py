import json

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


def debug_partial_cl_reward():
    model_name = "Qwen/Qwen3-0.6B"
    from transformers import AutoTokenizer
    import numpy as np
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    fn = "debug_0.json"
    fn = "debug_1.json"

    def helper(input_fn, input_tokenizer):
        with open(input_fn, "r") as f:
            # response', 'response_mask', 'classmate_mask', 'adv_mean', 'combined_advantages', 'advantages
            result_dict = json.load(f)
            main_response_token_ids = np.array(result_dict["response"])[np.array(result_dict["response_mask"]) == 1]
            main_response_str = input_tokenizer.decode(main_response_token_ids)
            cl_input_ids = np.array(result_dict["response"])[np.array(result_dict["classmate_mask"]) == 1]
            cl_input_str = input_tokenizer.decode(cl_input_ids)
            adv_mean = result_dict["adv_mean"]
            combined_advantages = result_dict["combined_advantages"]
            is_cl_combined_adv = np.array(combined_advantages)[np.array(result_dict["classmate_mask"]) == 1].tolist()
            is_non_cl_adv = np.array(result_dict["advantages"])[
                np.array(result_dict["classmate_mask"]) == 0].tolist()

            is_cl_advantages = np.array(result_dict["advantages"])[np.array(result_dict["classmate_mask"]) == 1].tolist()

            first_non_cl_token = input_tokenizer.decode(main_response_token_ids[int(sum(result_dict["classmate_mask"]) + 1)].item())
            first_non_cl_token_adv = result_dict["advantages"][int(sum(result_dict["classmate_mask"]) + 1)]
            last_cl_token_adv = result_dict["advantages"][int(sum(result_dict["classmate_mask"]))]
            breakpoint()

    # helper(fn, tokenizer)
    helper("debug_1.json", tokenizer)


if __name__ == "__main__":
    debug_partial_cl_reward()