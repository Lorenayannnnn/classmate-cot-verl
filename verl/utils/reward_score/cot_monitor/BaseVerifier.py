
from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List
import numpy as np

logger = logging.getLogger(__name__)


class BaseVerifier(ABC):
    """
    Base class for verifiers that support different reward types.

    Supports:
    - Binary rewards: reports TP, TN, FP, FN, precision, recall, F1
    - Non-binary rewards: reports MSE
    """

    def __init__(self, reward_type: str = "binary"):
        """
        Initialize the verifier with a reward type.

        Args:
            reward_type: Either "binary" or "non_binary"
        """
        if reward_type not in ["binary", "non_binary"]:
            raise ValueError(f"reward_type must be 'binary' or 'non_binary', got {reward_type}")
        self.reward_type = reward_type

    @abstractmethod
    def format_llm_judge_prompt(self, user_prompt, continuation, **kwargs):
        if self.reward_type == "binary":
            raise ValueError("Reward is binary; gt should be provided; should not use llm judge")
        else:
            raise NotImplementedError("LLM judge prompt formatting not implemented for this verifier. If you want to use LLM judge evaluation, please implement format_llm_judge_prompt in your verifier subclass.")

    @abstractmethod
    def parse_llm_judge_output(self, judge_response, judge_client_config):
        if self.reward_type == "binary":
            raise ValueError("Reward is binary; gt should be provided; should not use llm judge")
        else:
            raise NotImplementedError("LLM judge output parsing not implemented for this verifier. If you want to use LLM judge evaluation, please implement parse_llm_judge_output in your verifier subclass.")

    @abstractmethod
    def create_monitor_message(self, doc: Dict[str, Any]) -> str:
        """
        Create a monitor message for evaluation.

        Args:
            doc: Document containing question and metadata

        Returns:
            Formatted monitor message string
        """
        raise NotImplementedError("Monitor message creation not implemented for this verifier. If you want to use monitor-based evaluation, please implement create_monitor_message in your verifier subclass.")

    @abstractmethod
    def parse_monitor_output(self, monitor_response, monitor_config):
        raise NotImplementedError("Monitor output parsing not implemented for this verifier. If you want to use monitor-based evaluation, please implement parse_monitor_output in your verifier subclass.")

    def compute_metrics(self, predictions: List[float], ground_truths: List[float]) -> Dict[str, float]:
        """
        Compute metrics based on reward type.

        Args:
            predictions: List of predicted values
            ground_truths: List of ground truth values

        Returns:
            Dictionary of metric names to values
        """
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        if self.reward_type == "binary":
            return self._compute_binary_metrics(predictions, ground_truths)
        else:  # non_binary
            return self._compute_non_binary_metrics(predictions, ground_truths)

    @staticmethod
    def _compute_binary_metrics(predictions: np.ndarray, ground_truths: np.ndarray) -> Dict[str, float]:
        """
        Compute binary classification metrics.

        Returns:
            Dict with tp, tn, fp, fn, precision, recall, f1
        """
        # Ensure binary values
        predictions = (predictions > 0.5).astype(int)
        ground_truths = (ground_truths > 0.5).astype(int)

        tp = np.sum((predictions == 1) & (ground_truths == 1))
        tn = np.sum((predictions == 0) & (ground_truths == 0))
        fp = np.sum((predictions == 1) & (ground_truths == 0))
        fn = np.sum((predictions == 0) & (ground_truths == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "tp": float(tp),
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    @staticmethod
    def _compute_non_binary_metrics(predictions: np.ndarray, ground_truths: np.ndarray) -> Dict[str, float]:
        """
        Compute non-binary regression metrics.

        Returns:
            Dict with mse, rmse, mae
        """
        prediction_mean = np.mean(predictions)
        ground_truth_mean = np.mean(ground_truths)
        mse = np.mean((predictions - ground_truths) ** 2)
        # rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - ground_truths))

        return {
            "mse": float(mse),
            "prediction_mean": float(prediction_mean),
            "ground_truth_mean": float(ground_truth_mean),
            # "rmse": float(rmse),
            # "mae": float(mae)
        }

    @abstractmethod
    def compute_score(self, continuation: str, doc: Any) -> Dict[str, Any]:
        if self.reward_type == "binary":
            raise NotImplementedError("Should be implemented in subclass.")
        else:
            raise NotImplementedError("Should use llm judge; this function should not be called.")
        # """
        # Compute the score for a continuation.
        #
        # Args:
        #     continuation: The model's generated text
        #     gt: Ground truth value
        #     debug: Whether to output debug information
        #     **kwargs: Additional arguments
        #
        # Returns:
        #     Dict containing score and extracted_solution
        # """
        # try:
        #     res = self.extract_answer(continuation)
        #
        #     if self.reward_type == "binary":
        #         score = 1.0 if res["answer"] == gt else 0.0
        #     else:
        #         # For non-binary, we might need to parse both answer and gt as floats
        #         try:
        #             answer_val = float(res["answer"])
        #             gt_val = float(gt)
        #             # For non-binary, we could return the actual value or a normalized score
        #             # Here we'll return the actual predicted value
        #             score = answer_val
        #         except ValueError:
        #             score = 0.0
        #
        #     return {
        #         "score": score,
        #         "extracted_solution": res["answer"],
        #         "answer_format_correct": res.get("answer_format_correct", 1.0)
        #     }
        # except Exception as e:
        #     return {
        #         "score": 0.0,
        #         "extracted_solution": f"empty result (probably didn't end with </think>)" if continuation is None else f"ERROR: {str(e)}",
        #         "answer_format_correct": 0.0
        #     }

    @abstractmethod
    def process_doc(self, doc):
        """
        Used only in my_eval
        """
        raise NotImplementedError("Should be implemented in subclass.")


def _normalize_data_sources(data_source) -> List[str]:
    """Normalize data_source into a list of strings."""
    if data_source is None:
        return []
    if isinstance(data_source, str):
        return [data_source]
    if isinstance(data_source, np.ndarray):
        return [str(item) for item in data_source.ravel().tolist()]
    if isinstance(data_source, (list, tuple, set)):
        return [str(item) for item in data_source]
    return [str(data_source)]


def get_verifier(data_source=None) -> BaseVerifier:
    # TODO: if data_source is a list, currently assuming all elements are the same type
    """Get the appropriate verifier based on data_source.

    Args:
        data_source: Data source string or list/array to determine verifier type

    Returns:
        BaseVerifier instance appropriate for the data source
    """
    data_sources = _normalize_data_sources(data_source)
    data_source_joined = " ".join(data_sources)

    is_mmlu = "mmlu" in data_source_joined
    is_mmlu_pro = "mmlu_pro" in data_source_joined
    is_helpful = any("helpful_instructions" in source for source in data_sources)
    is_anthropic_hh_rlhf = any("anthropic_hh_rlhf" in source for source in data_sources)

    if is_mmlu:
        from verl.utils.reward_score.cot_monitor.MMLUSycophancyVerifier import MMLUSycophancyVerifier
        return MMLUSycophancyVerifier(do_pro=is_mmlu_pro, reward_type="binary")
    elif is_helpful:
        # from verl.utils.reward_score.cot_monitor.HelpfulInstructionVerifier import GeneralSycophancyVerifier
        # return GeneralSycophancyVerifier(reward_type="non_binary")
        from verl.utils.reward_score.cot_monitor.SycophancyVerifier import SycophancyVerifier
        return SycophancyVerifier(reward_type="non_binary")
    elif is_anthropic_hh_rlhf:
        from verl.utils.reward_score.cot_monitor.SycophancyVerifier import SycophancyVerifier
        return SycophancyVerifier(reward_type="non_binary")
    else:
        raise ValueError(f"Unsupported data_source for verifier: {data_source}")


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
    verifier = get_verifier(data_source=data_source)
    assert ground_truth is not None, "Ground truth must be provided for scoring. Otherwise use Tinker-based judge."

    # if ground_truth is None:
    #     # Verifier should score the output
    #     ground_truth, explanation = verifier.get_gt(user_message=extra_info["reward_model"]["raw_question_for_monitor"], continuation=solution_str)
    #     if ground_truth is None:
    #         logger.warning("⚠️Ground truth could not be obtained; defaulting to None; judge output")
    #         logger.warning(f"⚠️ {ground_truth}")
    #         ground_truth = 0.0      # no reward if we can't get a gt, but this is a fallback and ideally shouldn't happen
    #     score = ground_truth
    #     extracted_sol = solution_str
    # else:
    res = verifier.compute_score(solution_str, ground_truth)
    score = res["score"]
    extracted_sol = res["extracted_solution"]

    if return_dict:
        return {
            "score": score,
            "extracted_solution": extracted_sol
        }
    else:
        return score