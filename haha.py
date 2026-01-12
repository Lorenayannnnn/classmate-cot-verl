
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
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    given_string = """Given:

- Price per organic egg = **50 cents**
- Price of a tray of eggs with **30 eggs = $12**

1. **Calculate the cost per egg:**
   $$
   \\text{Cost per egg} = \frac{\\text{Price of a tray}}{\\text{Number of eggs in a tray}} = \frac{12}{30} = 0.40 \\text{ dollars}
   $$

   Convert to cents:
   $$
   0.40 \\text{ dollars} = 40 \\text{ cents}
   $$

2. **Calculate cost per individual egg:**
   $$
   \\text{Cost per egg (individual)} = 50 \\text{ cents}
   $$

3. **Find the difference between tray price and individual egg price:**
   $$"""

    token_ids_list = [22043, 1447, 12, 8483, 817, 17356, 18636, 284, 3070, 20, 15, 30191, 1019, 12, 8483, 315, 264, 34688, 315, 18805, 448, 3070, 18, 15, 18805, 284, 400, 16, 17, 56177, 16, 13, 3070, 47866, 279, 2783, 817, 18636, 25, 1019, 256, 400, 25046, 256, 1124, 1318, 90, 14940, 817, 18636, 92, 284, 1124, 37018, 35702, 1318, 90, 6972, 315, 264, 34688, 3417, 35702, 1318, 90, 2833, 315, 18805, 304, 264, 34688, 3417, 284, 1124, 37018, 90, 16, 17, 15170, 18, 15, 92, 284, 220, 15, 13, 19, 15, 1124, 1318, 90, 11192, 532, 256, 26107, 271, 256, 7169, 311, 30191, 510, 256, 400, 25046, 256, 220, 15, 13, 19, 15, 1124, 1318, 90, 11192, 92, 284, 220, 19, 15, 1124, 1318, 90, 30191, 532, 256, 26107, 271, 17, 13, 3070, 47866, 2783, 817, 3842, 18636, 25, 1019, 256, 400, 25046, 256, 1124, 1318, 90, 14940, 817, 18636, 320, 54877, 9139, 284, 220, 20, 15, 1124, 1318, 90, 30191, 532, 256, 26107, 271, 18, 13, 3070, 9885, 279, 6672, 1948, 34688, 3349, 323, 3842, 18636, 3349, 25, 1019, 256, 400, 25046, 256, 1124, 1318, 90, 50, 45751, 817, 18636, 92, 284, 220, 20, 15, 1124, 1318, 90, 30191, 92, 481, 220, 19, 15, 1124, 1318, 90, 30191, 92, 284, 220, 16, 15, 1124, 1318, 90, 30191, 532, 256, 26107, 271, 44364, 334, 16141, 66963, 2303, 14085, 198, 59, 79075, 90, 16, 15, 92, 1124, 1318, 90, 30191, 532, 14085, 151645, 151643]


    breakpoint()
    tokenizer.decode(token_ids_list[:179], skip_special_tokens=False)
    tokenizer.decode(token_ids_list[:180], skip_special_tokens=False)
    tokenizer.decode(token_ids_list[:181], skip_special_tokens=False)
    tokenizer.decode(token_ids_list[:182], skip_special_tokens=False)



if __name__ == "__main__":
    main()