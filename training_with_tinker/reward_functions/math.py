
from math_verify import parse, verify
from math_verify.errors import TimeoutException

def math_compute_score(
    solution_str,
    ground_truth,
    data_source=None,
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
    except TimeoutException as e:
        # logger.error("Timeout exception during math_verify.")
        score = 0
    except TimeoutError:
        # logger.error("Timeout error during math_verify.")
        score = 0
    except Exception as e:
        # logger.error(f"Error during math_verify: {e}")
        score = 0

    return {
        "score": score,
        "extracted_solution": extracted_sol
    }

    # if return_dict:
    #     return {
    #         "score": score,
    #         "extracted_solution": extracted_sol
    #     }
    # else:
    #     return score