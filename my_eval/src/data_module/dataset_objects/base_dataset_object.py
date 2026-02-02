import re
from abc import abstractmethod
from typing import List


# Reference: https://github.com/allenai/olmes
class BaseDatasetObject:
    def __init__(self):
        pass

    @abstractmethod
    def _fewshot_context(self):
        pass
        # return self.labeled_examples

    @abstractmethod
    def process_doc(self, doc):
        pass
        # """
        # HF Dataset class provides a map function that can pass an index to each doc with `with_indices=True`
        # """
        # # question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
        # # answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72
        # if "short_answer" in doc:
        #     short_answer = doc["short_answer"]
        # else:
        #     short_answer = doc["answer"].split("####")[-1].strip()
        #     doc["short_answer"] = short_answer
        # gold_cot = self.normalize_answer_str(doc, doc["answer"])
        # query = "Question: " + doc["question"] + "\nAnswer:"
        # cleaned_short_answer = self._clean_short_answer(short_answer)
        # out_doc = {
        #     "id": index,
        #     "question": doc["question"],
        #     "query": query,
        #     "answer": doc["answer"],
        #     "short_answer": short_answer,
        #     "cleaned_short_answer": cleaned_short_answer,
        #     "choices": [gold_cot],  # for perplexity eval
        # }
        # out_doc = self.apply_prompt_template(
        #     out_doc,
        # )
        # return out_doc

    @abstractmethod
    def apply_classmate_prompt_template(self, doc):
        pass

    @abstractmethod
    def extract_main_answer(self, continuation):
        pass

    @abstractmethod
    def extract_classmate_answer(self, continuation):
        pass

    def remove_last_step(self, data):
        main_pred = data["main_CoT"]
        keep_rate = 0.8  # TODO
        _SPLIT_WS = re.compile(r"\S+|\s+")
        num_to_keep = int(len(main_pred.split()) * keep_rate)  # counts non-whitespace tokens
        if num_to_keep <= 0:
            return main_pred
        kept = []
        count = 0
        for m in _SPLIT_WS.finditer(main_pred):
            t = m.group(0)
            # whitespace tokens start with a whitespace char
            if t[0].isspace():
                if count > 0:  # optional: avoid leading whitespace if truncating from start
                    kept.append(t)
            else:
                count += 1
                if count > num_to_keep:
                    break
                kept.append(t)

        return "".join(kept)

    @abstractmethod
    def compare_pred_and_gt(self, continuation: str, gt: str) -> bool:
        pass