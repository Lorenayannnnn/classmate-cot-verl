
import logging
from math_verify import parse, verify
from math_verify.errors import TimeoutException

logger = logging.getLogger(__name__)

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
    # if data_source == "MATH" or data_source == "gsm8k" or data_source == "deepscaler" or data_source == "aime":
    extracted_sol = None
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        # solution_str = parse(solution_str, parsing_timeout=None)
        # ground_truth = parse(ground_truth, parsing_timeout=None)
        # score = 1 if verify(solution_str, ground_truth, timeout_seconds=None) else 0
        extracted_sol = parse(solution_str)
        ground_truth = parse(ground_truth_boxed)
        score = 1 if verify(extracted_sol, ground_truth) else 0
    except TimeoutException | TimeoutError as e:
        # print("Timeout exception during math_verify.")
        logger.error("Timeout exception during math_verify.")
        score = 0
    except Exception as e:
        # print("Error during math_verify:", e)
        logger.error(f"Error during math_verify: {e}")
        score = 0

    if return_dict:
        return {
            "score": score,
            "extracted_solution": extracted_sol
        }
    else:
        return score
