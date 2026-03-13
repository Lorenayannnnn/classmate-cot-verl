"""BaseBackend wrapper that delegates to a Ray ScoringJudgeWorker.

Usage:
    # Before init_workers:
    backend = ScoringJudgeWorkerBackend()
    reward_fn = NaiveRewardManager(..., llm_judge_backend=backend)

    # After trainer.init_workers():
    backend.set_worker_group(trainer.scoring_judge_wg)
"""

import numpy as np

from verl import DataProto
from verl.workers.reward_model.judge_backend import BaseBackend


class ScoringJudgeWorkerBackend(BaseBackend):
    """Bridges the BaseBackend interface to a Ray ScoringJudgeWorker group.

    worker_group is None until set_worker_group() is called after init_workers().
    """

    def __init__(self):
        self.worker_group = None

    def set_worker_group(self, worker_group):
        """Set the RayWorkerGroup handle. Called after trainer.init_workers()."""
        self.worker_group = worker_group

    def prepare_input(self, verifier, question: str, output: str):
        """Return (question, output) directly; the worker handles formatting."""
        return (question, output)

    def run_batch(self, inputs: list) -> list:
        assert self.worker_group is not None, (
            "ScoringJudgeWorkerBackend.worker_group is None; "
            "call set_worker_group() after trainer.init_workers()"
        )
        n = len(inputs)
        questions = np.array(
            [item[0] if item is not None else None for item in inputs], dtype=object
        )
        outputs = np.array(
            [item[1] if item is not None else None for item in inputs], dtype=object
        )
        data = DataProto(batch=None, non_tensor_batch={"questions": questions, "outputs": outputs})

        result_data = self.worker_group.score_batch(data)
        # ONE_TO_ALL dispatch returns a list when there are multiple colocated workers;
        # all workers compute identical results, so just take the first.
        scores = result_data.non_tensor_batch["scores"].tolist()
        extracted = result_data.non_tensor_batch["extracted_solutions"].tolist()

        return [
            {
                "score": scores[i],
                "extracted_solution": extracted[i],
                "judge_explanation": "vllm_scoring_worker",
            } if inputs[i] is not None else None
            for i in range(n)
        ]

    def close(self):
        pass
