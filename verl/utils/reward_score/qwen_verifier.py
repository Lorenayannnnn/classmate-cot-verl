import concurrent.futures


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
        score = 0.0
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        def _compute():
            """Helper function to compute score with parsing and verification."""
            parsed_output = parse(solution_str, parsing_timeout=None)
            parsed_ground_truth = parse(ground_truth_boxed, parsing_timeout=None)
            return verify(parsed_output, parsed_ground_truth, timeout_seconds=None)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_compute)
                score = future.result(timeout=10)  # 10 seconds timeout
        except concurrent.futures.TimeoutError:
            print("Timeout during math_verify.")
            score = 0
        except Exception as e:
            print("Error during math_verify:", e)
            score = 0

        if return_dict:
            return {
                "score": score,
                "extracted_solution": solution_str
            }
        else:
            return score

        # try:
        #     from math_verify import parse, verify
        #     solution_str = parse(solution_str, parsing_timeout=None)
        #     ground_truth = parse(ground_truth, parsing_timeout=None)
        #     score = 1 if verify(solution_str, ground_truth, timeout_seconds=None) else 0
        #     if return_dict:
        #         return {
        #             "score": score,
        #             "extracted_solution": solution_str
        #         }
        #     else:
        #         return score
        # except Exception as e:
        #     print("Error during math_verify:", e)
        #     if return_dict:
        #         return {
        #             "score": 0,
        #             "extracted_solution": None
        #         }
        #     else:
        #         return 0
    else:
        raise NotImplementedError(f"Data source {data_source} not supported in qwen_verifier.")
