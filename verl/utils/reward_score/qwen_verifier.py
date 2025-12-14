
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
    if data_source == "deepscaler" or data_source == "aime":
        from math_verify import parse, verify
        solution_str = parse(solution_str)
        ground_truth = parse(ground_truth)
        score = 1 if verify(solution_str, ground_truth) else 0
        if return_dict:
            return {
                "score": score,
                "extracted_solution": solution_str
            }
        else:
            return score
    else:
        raise NotImplementedError(f"Data source {data_source} not supported in qwen_verifier.")
