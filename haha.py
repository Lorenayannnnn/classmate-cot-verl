import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    from openai import OpenAI
    client = OpenAI(
        base_url=f"http://141.148.74.128:8000/v1",        # TODO
        api_key="EMPTY",
    )
    def _generate_via_remote_api(model_name_or_path, batch_prompts: list) -> list:
        """Generate completions via remote vLLM API.

        Args:
            batch_prompts: List of input prompts

        Returns:
            List of generated completions
        """
        # Convert generation_config to API parameters
        vllm_api_params = {
            "model": model_name_or_path,
            "prompt": batch_prompts,
        }

        finish = False
        completions = []
        while not finish:
            try:
                print("🐛🐛🐛 Send request. Wait...")
                # breakpoint()
                response = client.completions.create(**vllm_api_params, timeout=60)
                # breakpoint()
                print(f"🐛🐛🐛 Response received: {response}")
                completions = [choice.text for choice in response.choices]
                print("🐛🐛🐛 Successful vLLM API completions:", completions)
                finish = True
            except Exception as e:
                print(f"Error checking model readiness: {e}")
                time.sleep(5)

        return completions


    model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
    batch_prompts = [
        "Tell me a joke about cats. Tell me a joke about cats. Tell me a joke about cats. Tell me a joke about cats. Tell me a joke about cats. Tell me a joke about cats.",
        "Explain the theory of relativity in simple terms. Explain the theory of relativity in simple terms. Explain the theory of relativity in simple terms. Explain the theory of relativity in simple terms. Explain the theory of relativity in simple terms.",
        "John Hewitt is a NLP professor at Columbia University",
    ]
    completions = _generate_via_remote_api(model_name_or_path, batch_prompts)
    for i, prompt in enumerate(batch_prompts):
        print(f"Prompt: {prompt}")
        print(f"Completion: {completions[i]}")
        print("-" * 40)

def test():
    # def execute_code(solution_code, test_case_code, import_prompt, entry_point, check_correctness=False):
    #     """
    #     Execute the solution code + test code to verify executability.
    #     Optionally run assertion tests for correctness.
    #     """
    #     try:
    #         # Build full code
    #         full_code = import_prompt + "\n" + solution_code + "\n" + test_case_code
    #
    #         # First check: compilation
    #         compiled = compile(full_code, "<string>", "exec")
    #
    #         # Execute in isolated namespace
    #         env = {}
    #         exec(compiled, env)
    #
    #         # Optional: run the checker
    #         if check_correctness:
    #             # Build candidate function dynamically from entry_point
    #             candidate = eval(entry_point, env)
    #             # Expose candidate to test case code (so `check(candidate)` works)
    #             env["candidate"] = candidate
    #             if "check" not in env:
    #                 return False  # No checker found
    #             env["check"](candidate)
    #
    #         return True
    #
    #     except Exception as e:
    #         # Any exception → not executable
    #         # print("Execution error:", e)
    #         return False
    #
    #
    # import_prompt = "from typing import List"
    #
    # entry_point = "Solution().twoSum"
    #
    # solution_code = """class Solution:
    # def twoSum(self, nums: List[int], target: int) -> List[int]:
    #     d = {}
    #     for i, x in enumerate(nums):
    #         if (target - x) in d:
    #             return [d[target - x], i]
    #         d[x] = i
    # """
    #
    # test_case_code = """def check(candidate):
    # assert candidate(nums=[3, 3], target=6) == [0, 1]
    # assert candidate(nums=[-1, -2, -3, -4], target=-8) is None
    # assert candidate(nums=[1000000000, 1000000000], target=2000000000) == [0, 1]
    # """
    #
    # # Test both modes
    # result_correctness = execute_code(solution_code, test_case_code,
    #                                   import_prompt, entry_point,
    #                                   check_correctness=True)
    #
    # result_executable = execute_code(solution_code, test_case_code,
    #                                  import_prompt, entry_point,
    #                                  check_correctness=False)
    #
    # print("Executability test:", result_executable)
    # print("Correctness test:", result_correctness)

    idx_to_language_name = ["unknown_language", "python", "cpp", "python3", "java"]
    system_prompt = "You are a programmer. You will be given a coding question, the test inputs and outputs, and solution. Your task is to modify the code to pass all the test cases."
    query_prompt_template = """## Question:
        {description}

        ## Test Inputs and Outputs
        ```json
        {test_inputs_outputs}
        ```
        
        ## Code:
        ```{language}
        {solution}
        ```

        ## Format: Think step by step, and return test inputs and outputs and the code solution using json and {language} backticks in the following ordering and format:
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
        
        ## Answer:
        """

    test_entry = {
        "description": "Andi and Budi were given an assignment to tidy up their bookshelf of n books. Each book is represented by the book title — a string s_i numbered from 1 to n, each with length m. Andi really wants to sort the book lexicographically ascending, while Budi wants to sort it lexicographically descending.\n\nSettling their fight, they decided to combine their idea and sort it asc-desc-endingly, where the odd-indexed characters will be compared ascendingly, and the even-indexed characters will be compared descendingly.\n\nA string a occurs before a string b in asc-desc-ending order if and only if in the first position where a and b differ, the following holds:\n\n  * if it is an odd position, the string a has a letter that appears earlier in the alphabet than the corresponding letter in b; \n  * if it is an even position, the string a has a letter that appears later in the alphabet than the corresponding letter in b. \n\nInput\n\nThe first line contains two integers n and m (1 ≤ n ⋅ m ≤ 10^6).\n\nThe i-th of the next n lines contains a string s_i consisting of m uppercase Latin letters — the book title. The strings are pairwise distinct.\n\nOutput\n\nOutput n integers — the indices of the strings after they are sorted asc-desc-endingly.\n\nExample\n\nInput\n\n\n5 2\nAA\nAB\nBB\nBA\nAZ\n\n\nOutput\n\n\n5 2 1 3 4\n\nNote\n\nThe following illustrates the first example.\n\n<image>",
        "test_inputs_outputs": {'input': ['5 2\nAA\nAB\nBB\nBA\nAZ\n'], 'output': ['5 2 1 3 4 \n']},
        "language": idx_to_language_name[2],
        "incorrect_solution": "#include <bits/stdc++.h>\nusing namespace std;\nmt19937 rnd(time(0));\nconst long long inf = 0x3f3f3f3f3f3f3f3fLL;\nlong long N = 3e5 + 10;\nconst long long MOD = 1e9 + 7;\nint32_t main() {\n  ios::sync_with_stdio(false);\n  cin.tie(0);\n  long long n, m;\n  cin >> n >> m;\n  string s;\n  vector<pair<long long, long long>> a;\n  for (long long i = 0; i < n; i++) {\n    cin >> s;\n    long long c = 0;\n    for (long long j = 0; j < m; j++) {\n      if (j & 1)\n        c -= (s[j] - 'A');\n      else\n        c += (s[j] - 'A');\n    }\n    a.emplace_back(c, i);\n  }\n  sort(a.begin(), a.end());\n  for (auto i : a) cout << i.second + 1 << ' ';\n  cout << '\\n';\n  return 0;\n}\n"
    }

    input_prompt = query_prompt_template.format(
        description=test_entry["description"],
        test_inputs_outputs=test_entry["test_inputs_outputs"],
        language=test_entry["language"],
        solution=test_entry["incorrect_solution"],
    )

    model_name = "allenai/OLMo-2-0425-1B-DPO"
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    input_ids = torch.tensor([tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ], add_generation_prompt=True, tokenize=True)])

    tokenized_input = {
        "input_ids": input_ids.to("cuda"),
        "attention_mask": torch.ones_like(input_ids).to("cuda"),
    }

    while True:
        generation_output = model.generate(**tokenized_input, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9)
        generated_text = tokenizer.decode(generation_output[0][len(tokenized_input["input_ids"][0]):], skip_special_tokens=True)
        print(generated_text)
        breakpoint()


if __name__ == '__main__':
    # main()
    # dataset = load_dataset("deepmind/code_contests", split="test")
    # test = dataset[0]
    # test = dataset[1]
    # print(dataset[0])
    # breakpoint()

    test()