import re

def extract_code(language, completion):
    solution = None
    if f"```{language}" in completion:
        solution = completion.split(f"```{language}")[-1].split("```")[0]
    elif "```" in completion:
        # Handle cases like ```\ncode\n```
        parts = completion.split("```")
        if len(parts) >= 2:
            solution = parts[1]
            # Remove potential language specifier like 'python\n'
            if "\n" in solution:
                first_line, rest = solution.split("\n", 1)
                if first_line.strip().isalpha():  # Simple check for language name
                    solution = rest

    return solution

# Reference: https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/__init__.py#L74
def check_correctness(sandbox_fusion_url, completion, test_inputs_outputs, language, concurrent_semaphore=None, memory_limit_mb=None):
    if sandbox_fusion_url:
        from . import sandbox_fusion

        # score: pass rate (passed / total_test_case_num)
        # final_metadata: some string description for the error. e.g. [{"error": "Invalid test_cases JSON format"}]
        score, final_metadata = sandbox_fusion.compute_score(
            sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, completion, test_inputs_outputs, language
        )
    else:
        assert language=="python", f"When Sandbox Fusion is not available, PrimeCode is used, which only supports python, but got {language}."
        # If no sandbox URL is provided, fall back to prime_code or raise error
        from . import prime_code

        # Assuming prime_code doesn't need the URL
        score, final_metadata = prime_code.compute_score(completion, test_inputs_outputs)
    return score, final_metadata

def compute_score(response_str, correct_sols, incorrect_sols, gt_test_cases, language, sandbox_fusion_url):
    """
    Args:
        response_str: Expected output format:
            ### Reasoning
            {{step_by_step_reasoning}}

            ### Test Inputs and Outputs
            ```json
            {{"input": [...], "output": [...]}}
            ```

            ### Code
            ```{language}
            {{code}}
            ```
        correct_sols: ["correct solution 1", "correct solution 2", ...]
        incorrect_sols: ["incorrect solution 1", "incorrect solution 2", ...]
        gt_test_cases: {"input": [...], "output": [...]}
        language: one of ["unknown_language", "python", "cpp", "python3", "java"]
        sandbox_fusion_url: URL string for sandbox fusion service
    """
    section_hash = "###"
    # reason_section_name = "### Reasoning"
    test_case_section_name = "### Test Inputs and Outputs"
    test_input_output_regex = rf"```json(.*?)```"
    code_section_name = "### Code"

    # reasoning_str = reason_section_name + "\n" + response_str.split(reason_section_name)[-1].split(test_case_section_name)[0].strip()

    test_inputs_outputs_by_regex = re.findall(test_input_output_regex, response_str, flags=re.DOTALL)
    if len(test_inputs_outputs_by_regex) == 0:
        # Fall back to using section name to split
        generated_inputs_outputs_str = response_str.split(test_case_section_name)[-1].split(code_section_name)[0].strip()
    else:
        generated_inputs_outputs_str = test_inputs_outputs_by_regex[0].strip()

    # Extract code and format it for sandbox execution
    extracted_code = response_str
    if code_section_name in response_str:
        extracted_code = response_str.split(code_section_name)[-1].split(section_hash)[0].strip()
    tmp_code = extract_code(language, extracted_code)
    if tmp_code is not None:
        extracted_code = "```" + language + "\n" + tmp_code + "\n```"

    result = {
        "generated_test_pass_correct_sol": 0.0,
        "generated_test_pass_incorrect_sol": 0.0,
        "generated_code_pass_gt_test": 0.0,
        "generated_code_pass_generated_test": 0.0,
    }

    # Run correctness check on generated test cases
    try:
        generated_inputs_outputs_str = eval(generated_inputs_outputs_str)
        # check_correctness(sandbox_fusion_url, completion, test_inputs_outputs, language, concurrent_semaphore=None, memory_limit_mb=None):
        # 1. Good test cases should pass ALL correct solutions: the higher the better
        total_passed_correct_sol = 0
        for correct_sol in correct_sols:
            score, _ = check_correctness(sandbox_fusion_url, correct_sol, generated_inputs_outputs_str, language)
            total_passed_correct_sol += 1 if score == 1.0 else 0
        result["generated_test_pass_correct_sol"] = total_passed_correct_sol / len(correct_sols) if len(correct_sols) > 0 else 0.0

        # 2. Good test cases should NOT pass ANY incorrect solutions: the closer to 0 the better
        total_passed_incorrect_sol = 0
        for incorrect_sol in incorrect_sols:
            score, _ = check_correctness(sandbox_fusion_url, incorrect_sol, generated_inputs_outputs_str, language)
            total_passed_incorrect_sol -= 1 if score != 0.0 else 0
        result["generated_test_pass_incorrect_sol"] = total_passed_incorrect_sol / len(incorrect_sols) if len(incorrect_sols) > 0 else 0.0

        # 3. Correctness: whether the generated code passes the generated test cases
        result["generated_code_pass_generated_test"], _ = check_correctness(sandbox_fusion_url, extracted_code, generated_inputs_outputs_str, language)
    except Exception:
        print(f"Failed to eval test_inputs_outputs_str: {generated_inputs_outputs_str}")

    # 4. Correctness: whether the generated code passes the ground truth test cases
    result["generated_code_pass_gt_test"], _ = check_correctness(sandbox_fusion_url, extracted_code, gt_test_cases, language)

    result["score"] = result["generated_code_pass_generated_test"]
    return result