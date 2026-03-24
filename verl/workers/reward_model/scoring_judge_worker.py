"""Ray Worker for HuggingFace sequence-classification reward-model scoring.

Runs AutoModelForSequenceClassification inside a dedicated Ray actor (which has
GPU access), following the ClassmateWorker pattern of moving the model to GPU
for inference and back to CPU when idle.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class ScoringJudgeWorkerConfig:
    model_name_or_path: str
    torch_dtype: torch.dtype

    @staticmethod
    def dtype_from_str(dtype_str: str | None) -> torch.dtype:
        """Convert a dtype string (e.g. 'bfloat16') to a torch.dtype. Defaults to bfloat16."""
        dtype = _DTYPE_MAP.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unknown llm_judge_backend_dtype: {dtype_str!r}. Choose from {list(_DTYPE_MAP)}")
        return dtype


class ScoringJudgeWorker(Worker):
    """Worker that runs a HuggingFace reward model as a dedicated Ray actor with GPU access.

    The model lives on CPU between calls and is moved to GPU only during score_batch,
    freeing GPU memory for the training workers while idle.

    Inputs (DataProto.non_tensor_batch):
        questions: np.ndarray[str | None]
        outputs:   np.ndarray[str | None]

    Outputs (DataProto.non_tensor_batch):
        scores:              np.ndarray[float | None]
        extracted_solutions: np.ndarray[str | None]
    """

    def __init__(self, config: ScoringJudgeWorkerConfig, role: str):
        super().__init__()
        self.config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name_or_path,
            torch_dtype=config.torch_dtype,
            attn_implementation="flash_attention_2",
            num_labels=1,
        )
        self._model.eval()
        # Start on CPU; moved to GPU only during score_batch
        self._model.to("cpu")
        print(f"🚀 Initialized ScoringJudgeWorker for {config.model_name_or_path} (device={self._device})")

    def __del__(self):
        del self._model
        del self._tokenizer

    def _onload_model(self):
        self._model.to(self._device)

    def _offload_model(self):
        self._model.to("cpu")
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """No-op: model already loaded in __init__."""
        pass

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def score_batch(self, data: DataProto) -> DataProto:
        """Score a batch of (question, output) pairs with the reward model."""
        questions = data.non_tensor_batch["questions"].tolist()
        outputs = data.non_tensor_batch["outputs"].tolist()
        n = len(questions)

        assert all(q is not None for q in questions), "All questions must be non-None"

        scores = []
        try:
            self._onload_model()

            formatted_texts = []
            for i in range(n):
                conv = [
                    {"role": "user", "content": questions[i]},
                    {"role": "assistant", "content": outputs[i] if outputs[i] is not None else ""},
                ]
                text = self._tokenizer.apply_chat_template(conv, tokenize=False)
                if self._tokenizer.bos_token and text.startswith(self._tokenizer.bos_token):
                    text = text[len(self._tokenizer.bos_token):]
                formatted_texts.append(text)

            tokenized = self._tokenizer(
                formatted_texts, return_tensors="pt", padding=True, truncation=False,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**tokenized).logits  # (batch, num_labels)

            for i in range(n):
                scores.append(float(logits[i, 0].item()))

        except Exception as e:
            print(f"❌ Error in ScoringJudgeWorker.score_batch: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            self._offload_model()

        assert len(scores) == n, f"Expected {n} scores, got {len(scores)}"

        return DataProto(batch=None, non_tensor_batch={
            "scores": np.array(scores, dtype=object),
            "extracted_solutions": np.array(outputs, dtype=object),
        })
