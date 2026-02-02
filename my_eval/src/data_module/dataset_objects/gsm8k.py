import re
from typing import List
from math_verify import parse, verify
from math_verify.errors import TimeoutException

from my_eval.src.data_module.dataset_objects.base_dataset_object import BaseDatasetObject


# Reference: https://github.com/allenai/olmes
class GSM8K(BaseDatasetObject):
    def __init__(self, wo_cot=False):
        super().__init__()
        self.wo_cot = wo_cot

        self.main_CoT_key_name = "main_CoT"
        self.gt_key_name = "gt"

        self.prompt_suffix = " Put your final answer within \\boxed{}."

    def apply_classmate_prompt_template(self, doc):
        if self.wo_cot:
            doc = self._process_doc_helper(doc)
            doc["classmate_query"] = doc["query"]
        else:
            doc["classmate_query"] = doc["question"] + self.prompt_suffix
        return doc

    def _process_doc_helper(self, doc, index=-1):
        """
        HF Dataset class provides a map function that can pass an index to each doc with `with_indices=True`
        """
        if "short_answer" in doc:
            short_answer = doc["short_answer"]
        else:
            short_answer = doc["answer"].split("####")[-1].strip()
            doc["short_answer"] = short_answer
        cleaned_short_answer = self._clean_short_answer(short_answer)

        query = doc["question"] + self.prompt_suffix

        out_doc = {
            "id": index,
            "question": doc["question"],
            "query": query,
            "answer": doc["answer"],
            "gt": short_answer,
            "cleaned_gt": cleaned_short_answer,
        }
        return out_doc

    def process_doc(self, doc):
        doc = self._process_doc_helper(doc)
        return doc

    def _add_spaces_around_operators_no_regex(self, _str):
        """Add spacing around special operators if it does not exist"""
        operators = {"+", "-", "*", "/", "="}
        result: List[str] = []
        for char in _str:
            if char in operators:
                if result and result[-1] != " ":
                    result.append(" ")
                result.append(char)
                result.append(" ")
            else:
                result.append(char)

        # Join the list and replace double spaces with single spaces
        return "".join(result).replace("  ", " ")


    def _normalize_answer_str(self, doc: dict, answer: str) -> str:
        """
        Convert the gold CoT to a more natural-appearing string to improve bpb calculation.

        Question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
        Original Answer: Janet has 16 eggs and uses 4 for baking and sells 3 for breakfast. Therefore, she makes 16 - 3 - 4 = <<16-3-4=9>>9 eggs sold, leading to a daily income of 9 * 2 = $<<9*2=22>>22.\n#### 22
        New Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmer’s market. So the answer is 18.
        """
        import re

        answer = re.sub(r"<<.*?>>", "", answer)
        answer = re.sub(r"\s+", " ", answer).strip()
        answer = re.split(r"####", answer)[0]
        answer = answer[0].capitalize() + answer[1:] if answer else answer
        answer = answer.strip()
        if not answer.endswith("."):
            answer += "."
        answer = answer + f" So the answer is {doc['short_answer']}."
        answer = self._add_spaces_around_operators_no_regex(answer)
        return answer


    def _clean_short_answer(self, continuation: str):
        # ANS_RE = re.compile(r"answer is (\-?[0-9\.\,]+).")
        # INVALID_ANS = "[invalid]"
        # Replace commas
        output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            return numbers[-1]
        else:
            return output

    def extract_main_answer(self, continuation: str, extract_cot=False):
        assert extract_cot is False, "extract cot deprecated"
        pred = parse(continuation)

        return pred

    def extract_numerical_value(self, continuation: str):
        parsed_result = parse(continuation)
        if len(parsed_result) == 0:
            return ""
        return parse(continuation)[-1]

    def extract_classmate_answer(self, continuation):
        return self.extract_numerical_value(continuation)

    def compare_pred_and_gt(self, pred, gt) -> bool:
        ground_truth_boxed = "\\boxed{" + gt + "}"
        try:
            extracted_sol = parse(pred)
            ground_truth = parse(ground_truth_boxed)
            score = 1 if verify(ground_truth, extracted_sol) else 0
        except TimeoutException | TimeoutError as e:
            # print("Timeout exception during math_verify.")
            score = 0
        except Exception as e:
            # print("Error during math_verify:", e)
            score = 0

        return score == 1