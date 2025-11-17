
def execute_code(solution_code, test_case_code, import_prompt, entry_point, check_correctness=False) -> bool:
    """
    Execute the solution code + test code to verify executability.
    Optionally run assertion tests for correctness.

    solution_code: str, the code defining the solution function/class
    test_case_code: str, the code defining the test cases (e.g., check function)
    import_prompt: str, any necessary import statements
    entry_point: str, the name of the function to test (e.g., "Solution().twoSum")
    check_correctness: bool, whether to run the test cases for correctness
    """
    try:
        # Build full code
        full_code = import_prompt + "\n" + solution_code + "\n" + test_case_code

        # First check: compilation
        compiled = compile(full_code, "<string>", "exec")

        # Execute in isolated namespace
        env = {}
        exec(compiled, env)

        # Optional: run the checker
        if check_correctness:
            # Build candidate function dynamically from entry_point
            candidate = eval(entry_point, env)
            # Expose candidate to test case code (so `check(candidate)` works)
            env["candidate"] = candidate
            if "check" not in env:
                return False  # No checker found
            env["check"](candidate)

        return True

    except Exception as e:
        # Any exception → not executable
        # print("Execution error:", e)
        return False


def compute_score(solution_str, ground_truth, entry_point, gt_test_case, import_prompt, test_case_format_score=1.0, correct_code_sol_score=1.0):
    """The scoring function for leetcode code and test case generation.
    This is checking
    1. if the test case is well formatted (i.e., can be executed)
    2. if the generated code can pass the generated test case (Test case might not be reliable! We are not checking correctness here!)

    Args:
        solution_str: the solution text. Expected format is:
            class Solution:
                def twoSum(self, nums: List[int], target: int) -> List[int]:

            # 2. Test case:
            def check(candidate):
                assert candidate(

            ```
        ground_truth: Completion from the LeetCode dataset
        gt_test_case: test case provided from the dataset (Not used for now)
        import_prompt: python package imports prompt provided from the dataset
        score: the score for the correct answer
    """

    # Use solution/test case start end to parse out solution and test case
    test_case_start_str = "# 2. Test case:"
    test_case_end_str = "```"

    score = {
        "test_case_format_score": 0.0,
        "correct_code_score": 0.0,
    }

    solution_code = solution_str.split(test_case_start_str)[0].strip()
    test_case_code = solution_str.split(test_case_start_str)[-1].split(test_case_end_str)[0].strip()

    # Check if test case is well formatted (NOT checking reliability of the test case)
    if execute_code(ground_truth, test_case_code, import_prompt, entry_point, check_correctness=False):
        score["test_case_format_score"] = test_case_format_score
        # Check if the generated code can pass the generated test case
        if execute_code(solution_code, test_case_code, import_prompt, entry_point, check_correctness=True):
            score["correct_code_score"] = correct_code_sol_score
    score["score"] = score["test_case_format_score"] + score["correct_code_score"]
    return score