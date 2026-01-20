import re


class MMLU:
    def __init__(self):
        self.answer_format_regex = "Therefore, the answer is \\(([A-D])\\)"

    def compute_score(self, continuation, gt):
        res = self._extract_answer(continuation)
        return {
            "score": 1.0 if res["answer"] == gt else 0.0,
            "extracted_sol": res["answer"],
            "answer_format_correct": res["answer_format_correct"],
        }

    def _extract_answer(self, continuation: str):
        answer_format_correct = 1.0
        answer_string = None
        # if self.task_config["metric_kwargs"].get("answer_regexes"):
        #     res = extract_answer(continuation, task_config=self.task_config)
        #     return res
        # if self.task_config["metric_kwargs"].get("answer_format_regex"):
        matches = re.findall(
            self.answer_format_regex, continuation
        )
        if matches:
            # Pick the last occurrence
            answer_string = matches[-1]
        else:
            answer_format_correct = 0.5  # No direct match

        if answer_string is None:
            backup_patterns = [
                (r"(?i)therefore,?\s*the\s*answer\s*is:?\s*\(?([A-D])\b", 0.5),
                (r"(?i)answer\s*is:?\s*\(?([A-D])\b", 0.5),
                (r"(?i)([A-D])\)?\s+is\s+correct", 0.5),
                (r"\(([A-D])\)", 0.25),  # Any parenthesized capital letter
                (r".*\b([A-D])\b", 0),  # Any stand-alone capital letter
            ]
            for pattern, score in backup_patterns:
                match = re.search(pattern, continuation)
                if match:
                    answer_string = match.group(1).upper()
                    answer_format_correct = 0
                    break
        # Remove any parentheses
        if answer_string is None:
            answer_string = ""
        answer = re.sub("\\(|\\)", "", answer_string)
        return {"answer": answer, "answer_format_correct": answer_format_correct}