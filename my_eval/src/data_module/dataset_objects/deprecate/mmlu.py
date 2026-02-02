import re
from typing import List, Optional

from lm_eval.tasks.hendrycks_math.utils import last_boxed_only_string
from lm_eval.tasks.minerva_math.utils import get_unnormalized_answer, normalize_final_answer, is_equiv, remove_boxed
from lm_eval.tasks.minerva_math.utils import (
    process_results as minerva_math_process_results,
)
from lm_eval.tasks.hendrycks_math.utils import is_equiv as hendrycks_is_equiv
from lm_eval.tasks.hendrycks_math.utils import (
    process_results as hendrycks_process_results,
)

from my_eval.src.data_module.dataset_objects.base_dataset_object import BaseDatasetObject
from my_eval.src.data_module.utils import make_mcq_prompt, extract_answer, make_choices_text


# Reference: https://github.com/allenai/olmes
class MMLU(BaseDatasetObject):
    def __init__(self, subject):
        super().__init__()
        self.prompt_template = {
            "main_model": "",
            # "classmate_cot": "Chain of thought:\n{input_cot}\n\nBased only on this reasoning, choose from the following options and return only the option letter:\n{choice_text}\nAnswer: "
            "classmate_cot": "Question:\n{question}\nChain of thought:\n{input_cot}\n\nBased only on this reasoning, choose from the following options and return only the option letter:\n{choice_text}\nAnswer: "
        }

        self.main_CoT_key_name = "main_CoT"
        self.gt_key_name = "gt"

        self.subset = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
                       'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics',
                       'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics',
                       'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts',
                       'high_school_biology', 'high_school_chemistry', 'high_school_computer_science',
                       'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
                       'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics',
                       'high_school_physics', 'high_school_psychology', 'high_school_statistics',
                       'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality',
                       'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management',
                       'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios',
                       'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law',
                       'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies',
                       'sociology', 'us_foreign_policy', 'virology', 'world_religions']

        self.task_config = {
            "native_id_field": "question_id",
            "primary_metric": "exact_match",
            "split": "test",
            "generation_kwargs": {
                "max_gen_toks": 512,
                "do_sample": False,
                "temperature": 0.0,
                "stop_sequences": ["Question:"],
            },
            "chat_overrides": {
                "generation_kwargs": {
                    "stop_sequences": ["Question:"],
                },
                "context_kwargs": {
                    "label_format": None,  # Defaults to " A.", could be " (A)" etc
                    "fewshot_as_multiturn": False,
                },
            },
            "context_kwargs": {
                "description": 'Answer the following multiple-choice question by giving the correct answer letter in parentheses. Provide CONCISE reasoning for the answer, and make sure to finish the response with "Therefore, the answer is (ANSWER_LETTER)" where (ANSWER_LETTER) is one of (A), (B), (C), (D).\n\n'
            },
            "metric_kwargs": {
                "answer_format_regex": "Therefore, the answer is \\(([A-D])\\)",
            },
        }

        self.main_CoT_key_name = "main_CoT"
        self.gt_key_name = "gt_mcq_str"  # For filtering out last step in CoT

    def process_doc(self, doc, index=-1):
        num_choices = len(doc["choices"])
        choice_labels = ["A", "B", "C", "D", "E"][:num_choices]
        label_format = self.task_config["context_kwargs"].get("label_format", " A.")
        query = make_mcq_prompt(
            doc["question"], doc["choices"], question_prefix=self.task_config["context_kwargs"]["description"], answer_prefix="", label_format=label_format
        )
        out_doc = doc.copy()
        out_doc["query"] = query
        out_doc["choices"] = choice_labels
        out_doc["gt"] = choice_labels[doc["answer"]]
        out_doc["gt_mcq_str"] = doc["choices"][doc["answer"]]
        out_doc["choice_text"] = make_choices_text(doc["choices"], label_format=self.task_config["context_kwargs"].get("label_format", " A."))

        return out_doc

    def extract_main_answer(self, continuation: str, extract_cot=False):
        """
        This is pre-processing step for this task on the generated continuation from the request.
        """
        answer_format_correct = 1.0
        answer_string = None
        if self.task_config["metric_kwargs"].get("answer_regexes"):
            res = extract_answer(continuation, task_config=self.task_config)
            return res
        if self.task_config["metric_kwargs"].get("answer_format_regex"):
            matches = re.findall(
                self.task_config["metric_kwargs"]["answer_format_regex"], continuation
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
        # Split CoT by self.task_config["metric_kwargs"]["answer_format_regex"]
        tmp_cot = continuation.rsplit("Therefore, the answer is", 1)[0].strip()
        tmp_cot, tmp_pred = tmp_cot, answer
        return tmp_cot, tmp_pred


    def remove_last_step(self, data):
        finish = False
        cot_exclude_last_step = data[self.main_CoT_key_name].strip()
        max_remove_step_num = 2
        # print(data["question"])
        # print(cot_exclude_last_step)
        # print(data[self.gt_key_name])
        # breakpoint()
        while not finish and max_remove_step_num > 0:  # keep removing last step until the last line does not contain the final answer
            split_by_last_newline = cot_exclude_last_step.rsplit("\n", 1)
            if len(split_by_last_newline) == 1 or not data[self.gt_key_name] in split_by_last_newline[1]:
                finish = True
            else:
                cot_exclude_last_step = split_by_last_newline[0].strip()
            max_remove_step_num -= 1
        return cot_exclude_last_step

    def apply_classmate_prompt_template(self, doc):
        doc["query"] = self.prompt_template["classmate_cot"].format(
            question=doc["question"],
            input_cot=doc["main_cot_exclude_last_step"],
            choice_text=doc["choice_text"],
        )
        return doc

    def extract_classmate_answer(self, continuation):
        return continuation.strip()

    def compare_pred_and_gt(self, pred: str, gt: str) -> bool:
        # for processing classmate pred: exact_match
        return self.extract_main_answer(pred)[1] == gt
