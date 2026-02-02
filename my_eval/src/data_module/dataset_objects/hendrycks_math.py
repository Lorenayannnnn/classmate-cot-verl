import re

from my_eval.src.data_module.dataset_objects.base_dataset_object import BaseDatasetObject
from math_verify import parse, verify
from math_verify.errors import TimeoutException


class HendrycksMath(BaseDatasetObject):
    def __init__(self, wo_cot=False):
        super().__init__()

        self.main_CoT_key_name = "main_CoT"
        self.gt_key_name = "gt"
        self.wo_cot = wo_cot

        self.prompt_suffix = " Put your final answer within \\boxed{}."

    def process_doc(self, doc, index=-1):
        solution = doc["solution"]
        # query = doc["problem"] + " " + self.prompt_template
        query = doc["problem"] + self.prompt_suffix
        out_doc = {
            "id": index,
            "question": doc["problem"],
            "query": query,
            "level": doc.get("level"),
            "type": doc.get("type"),
            "gt": self.extract_main_answer(solution),
        }

        return out_doc

    def extract_main_answer(self, continuation: str):
        parsed_result = parse(continuation)
        if len(parsed_result) == 0:
            return ""
        return parse(continuation)[-1]

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

    def apply_classmate_prompt_template(self, doc):
        if self.wo_cot:
            doc = self.process_doc(doc)
            doc["classmate_query"] = doc["query"]
        else:
            doc["classmate_query"] = doc["question"] + self.prompt_suffix
        return doc

    def extract_classmate_answer(self, continuation):
        return self.extract_main_answer(continuation)

