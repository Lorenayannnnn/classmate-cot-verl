import re
from typing import List
from math_verify import parse, verify
from math_verify.errors import TimeoutException

from my_eval.src.data_module.dataset_objects.base_dataset_object import BaseDatasetObject


# Reference: https://github.com/allenai/olmes
class AIME(BaseDatasetObject):
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

        query = doc["problem"] + self.prompt_suffix

        out_doc = {
            "id": index,
            "question": doc["problem"],
            "query": query,
            "answer": doc["answer"],
            "gt": doc["answer"],
        }
        return out_doc

    def process_doc(self, doc):
        doc = self._process_doc_helper(doc)
        return doc

    def extract_main_answer(self, continuation: str, extract_cot=False):
        assert extract_cot is False, "extract cot deprecated"
        pred = self.extract_numerical_value(continuation)

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