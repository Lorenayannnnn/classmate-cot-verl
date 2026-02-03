
import logging
import re

logger = logging.getLogger(__name__)

class MMLU:
    def __init__(self, do_pro=False):
        if do_pro:
            self.answer_format_regex = fr"Therefore, the answer is \(([A-J])\)[.]*"
            self.backup_patterns = [
                (r"(?i)therefore,?\s*the\s*answer\s*is:?\s*\(?([A-J])\b", 0.5),
                (r"(?i)answer\s*is:?\s*\(?([A-J])\b", 0.5),
                (r"(?i)([A-J])\)?\s+is\s+correct", 0.5),
                (r"\(([A-J])\)", 0.25),  # Any parenthesized capital letter
                (r".*\b([A-J])\b", 0),  # Any stand-alone capital letter
            ]
        else:
            self.answer_format_regex = fr"Therefore, the answer is \(([A-D])\)[.]*"
            self.backup_patterns = [
                (r"(?i)therefore,?\s*the\s*answer\s*is:?\s*\(?([A-D])\b", 0.5),
                (r"(?i)answer\s*is:?\s*\(?([A-D])\b", 0.5),
                (r"(?i)([A-D])\)?\s+is\s+correct", 0.5),
                (r"\(([A-D])\)", 0.25),  # Any parenthesized capital letter
                (r".*\b([A-D])\b", 0),  # Any stand-alone capital letter
            ]

    # def clean_CoT(self, text, main_cot_keep_ratio=1.0):
    #     """
    #     Remove all trailing occurrences of the answer pattern from the text.
    #     Searches from the end backwards to remove only answers at the very end.
    #     """
    #     if main_cot_keep_ratio < 1:
    #         cleaned_text = text[:int(len(text) * main_cot_keep_ratio)]
    #     else:
    #         cleaned_text = text
    #         while True:
    #             # Find all matches
    #             all_matches = list(re.finditer(self.answer_format_regex, cleaned_text))
    #             if not all_matches:
    #                 break
    #
    #             # Check the last match
    #             last_match = all_matches[-1]
    #             after_match = cleaned_text[last_match.end():]
    #
    #             if after_match.strip() == "":
    #                 # This is a trailing occurrence, remove it
    #                 cleaned_text = cleaned_text[:last_match.start()].rstrip()
    #             else:
    #                 # Last match has content after it, stop removing
    #                 break
    #
    #     return cleaned_text

    def process_doc(self, doc):
        """
        Used only in my_eval
        """
        out_doc = {
            "question": doc["prompt"][0]["content"],
            "query": doc["prompt"][0]["content"],
            "actual_answer": doc["reward_model"]["actual_ground_truth"],
            "gt": doc["reward_model"]["ground_truth"],
            "raw_question_for_monitor": doc["reward_model"]["raw_question_for_monitor"],
            "hint_str_for_monitor": doc["reward_model"]["hint_str_for_monitor"]
        }
        return out_doc

    def compute_score(self, continuation, gt, debug=False):
        try:
            res = self._extract_answer(continuation)
            return {
                "score": 1.0 if res["answer"] == gt else 0.0,
                "extracted_solution": res["answer"],
                "answer_format_correct": res["answer_format_correct"]
            }
        except Exception as e:
            return {
                "score": 0.0,
                "extracted_solution": f"empty result (probably didn't end with </think>" if continuation is None else f"ERROR: {str(e)}",
                "answer_format_correct": 0.0
            }

    def _extract_answer(self, continuation: str):
        answer_format_correct = 1.0
        answer_string = None
        pre_answer_text = continuation if continuation is not None else ""
        # if self.task_config["metric_kwargs"].get("answer_regexes"):
        #     res = extract_answer(continuation, task_config=self.task_config)
        #     return res
        # if self.task_config["metric_kwargs"].get("answer_format_regex"):

        matches = list(re.finditer(self.answer_format_regex, continuation))
        if matches:
            # Pick the last occurrence (which is now the real last one)
            last_match = matches[-1]
            answer_string = last_match.group(1)
            pre_answer_text = continuation[:last_match.start()]
        else:
            answer_format_correct = 0.0  # No direct match

        if answer_string is None:
            for pattern, score in self.backup_patterns:
                match = re.search(pattern, continuation)
                if match:
                    answer_string = match.group(1).upper()
                    pre_answer_text = continuation[:match.start()]
                    answer_format_correct = 0
                    break
        # Remove any parentheses
        if answer_string is None:
            answer_string = ""
        answer = re.sub("\\(|\\)", "", answer_string)

        return {
            "answer": answer,
            "answer_format_correct": answer_format_correct,
            "pre_answer_text": pre_answer_text,
        }

mmlu_verifier = MMLU(do_pro=False)

def compute_score(
    data_source,
    solution_str,
    ground_truth=None,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    method=None,
    return_dict=True,
    **kwargs,
):
    res = mmlu_verifier.compute_score(solution_str, ground_truth, debug=False)
    score = res["score"]
    extracted_sol = res["extracted_solution"]

    if return_dict:
        return {
            "score": score,
            "extracted_solution": extracted_sol
        }
    else:
        return score