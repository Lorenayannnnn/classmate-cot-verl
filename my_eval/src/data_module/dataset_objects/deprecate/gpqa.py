import random
import re

from my_eval.src.data_module.dataset_objects.base_dataset_object import BaseDatasetObject


class GPQA(BaseDatasetObject):
    def __init__(self):
        super().__init__()
        self.task_config = {
            "dataset_path": "Idavidrein/gpqa",
            "dataset_name": "gpqa_main",
            "split": "train",
            "native_id_field": "id",  # "Dataset auto index"
            "primary_metric": "exact_match",
            "fewshot_source": "Original:GPQA",
            "generation_kwargs": {
                "max_gen_toks": 1024,
                "do_sample": False,
                "temperature": 0.0,
                "stop_sequences": ["</s>"],
            },
            "context_kwargs": {
                "answer_shuffling_seed": 111,
            },
            "metric_kwargs": {},
            "chat_overrides": {
                "context_kwargs": {
                    "description": "Answer the following multiple choice question. The last line of your response should be of the following format: `The correct answer is: ($LETTER)` where LETTER is one of ABCD. Think step by step before answering.\n",
                    "assistant_prefix": "\nLet's think step by step:\n",
                    "fewshot_as_multiturn": True,
                },
            },
        }

        self.prompt_template = {
            "assistant_prefix": "\nLet's think step by step:\n",
            "main_model": "Answer the following multiple choice question. The last line of your response should be of the following format: `The correct answer is: ($LETTER)` where LETTER is one of ABCD. Think step by step before answering.\n",
            "classmate_cot": "Chain of thought:\n{input_cot}\n\nBased only on this reasoning, return only the numerical value of the answer in the following format: \nFinal Answer: <number>"
        }

        self.main_CoT_key_name = "main_CoT"
        self.gt_key_name = "gt_mcq_str"     # For filtering out last step in CoT

    def _preprocess(self, text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def process_doc(self, doc, index=-1):
        """
        HF Dataset class provides a map function that can pass an index to each doc with `with_indices=True`
        """
        choices = [
            self._preprocess(doc["Incorrect Answer 1"]),
            self._preprocess(doc["Incorrect Answer 2"]),
            self._preprocess(doc["Incorrect Answer 3"]),
            self._preprocess(doc["Correct Answer"]),
        ]
        random.Random(self.task_config["context_kwargs"]["answer_shuffling_seed"] + index).shuffle(
            choices
        )
        correct_answer_index = choices.index(self._preprocess(doc["Correct Answer"]))
        choice_labels = ["A", "B", "C", "D"]
        query = self.prompt_template["main_model"] + "Question: " + doc["Question"] + "\nChoices:\n"
        query += "".join([f" ({key}) {choice}\n" for key, choice in zip(choice_labels, choices)])
        query += self.prompt_template["assistant_prefix"]
        out_doc = {
            "id": index,
            "question": doc["Question"],
            "query": query,
            "choices": choices,
            "explanation": doc["Explanation"],
            "gt": choice_labels[correct_answer_index],
            "gt_mcq_str": choices[correct_answer_index],
            "canary_string": doc["Canary String"],
        }

        return out_doc

    def extract_main_answer(self, continuation: str, extract_cot=False):
        """
        This is pre-processing step for this task on the generated continuation from the request.
        """
        if extract_cot:
            tmp_pred = self._extract_answer(continuation)
            if "The correct answer is: " in continuation:
                tmp_cot = continuation.split("The correct answer is: ")[0].strip()
            else:
                tmp_cot = continuation.split(tmp_pred)[0].strip()
            return tmp_cot, tmp_pred
        else:
            return None, self._extract_answer(continuation)


    def _extract_answer(self, continuation: str):
        answer_format_correct = 1.0
        answer_string = None
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
            match = re.findall(r"\(([A-D])\)", continuation)
            match2 = re.search(r"(?<=The correct answer is: )([A-D])", continuation)
            if match:
                answer_string = match[-1]
            elif match2:
                answer_string = match2.group(1)
            else:
                answer_format_correct = 0.0
                # Grab last capital A-D stand-alone letter as last effort
                match = re.findall(".*\\b([A-D])\\b", continuation)
                if match:
                    answer_string = match[-1]
                else:
                    answer_string = ""
        # Remove any parentheses
        answer = re.sub("\\(|\\)", "", answer_string)
        # return {"answer": answer, "answer_format_correct": answer_format_correct}
        return answer

    def _clean_answer(self, answer_dict):
        answer = answer_dict["answer"]
        answer_new = re.sub("\\(|\\)", "", answer)
        answer_dict["answer"] = answer_new
        if answer_new != answer:
            answer_dict["answer_raw"] = answer
        return answer_dict


    def remove_last_step(self, data):
        finish = False
        cot_exclude_last_step = data[self.main_CoT_key_name].strip()
        while not finish:
            split_by_last_newline = cot_exclude_last_step.rsplit(".", 1)
            if len(split_by_last_newline) == 1 or not data[self.gt_key_name] in split_by_last_newline[1]:
                finish = True
            else:
                cot_exclude_last_step = split_by_last_newline[0].strip()
        return cot_exclude_last_step
