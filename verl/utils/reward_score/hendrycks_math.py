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

import re

from verl.utils.math_utils import hendrycks_math_extract_solution, is_equiv, hendrycks_is_equiv

_SOLUTION_CLIP_CHARS = 300


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
    # 🐛 strictly extracted answer: {answer}
    # 🐛 flexibly extracted answer: {hendrycks_math_extract_solution(solution_str=solution_str")}
    # """)

    if answer is None:
        return 0
    else:
        if is_equiv(answer, ground_truth) or hendrycks_is_equiv(answer, ground_truth):
            return score
        else:
            return 0
