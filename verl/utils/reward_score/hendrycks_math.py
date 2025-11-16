
from verl.utils.math_utils import hendrycks_math_extract_solution, is_equiv, hendrycks_is_equiv


def compute_score(solution_str, ground_truth, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = hendrycks_math_extract_solution(solution_str=solution_str)

    # print(f"""
    # 🐛 solution_str: {solution_str}
    # 🐛 ground_truth: {ground_truth}
    # 🐛 extracted answer: {hendrycks_math_extract_solution(solution_str=solution_str)}
    # 🐛 is_correct: {is_equiv(answer, ground_truth) or hendrycks_is_equiv(answer, ground_truth)}
    # """)

    if answer is None:
        return 0
    else:
        if is_equiv(answer, ground_truth) or hendrycks_is_equiv(answer, ground_truth):
            return score
        else:
            return 0
