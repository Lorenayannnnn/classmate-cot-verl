import os

from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForCausalLM


def upload_ckpts_to_huggingface():
    from pathlib import Path
    from huggingface_hub import HfApi, create_repo
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    USER_OR_ORG = "LorenaYannnnn"                 # <-- change if needed
    # ROOT = Path("outputs/classmate_cot_w_verl/grpo_olmo_1B_gsm8k_baseline")
    # REPO_NAME = "classmate-cot-baseline"  # single repo name

    # ROOT = Path("/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs_20251030/grpo_olmo_1B_with_classmate_llama")
    # REPO_NAME = "20251030-grpo_olmo_1B_with_classmate_llama"

    # ROOT = Path("/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs/grpo_olmo_1B_gsm8k_baseline_400000_episodes")
    # REPO_NAME = "20251106-grpo_olmo_1B_gsm8k_baseline_400000_episodes"

    # ROOT = Path("/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs/grpo_olmo_1B_with_classmate_llama_400000_episodes")
    # REPO_NAME = "20251106-grpo_olmo_1B_with_classmate_llama_400000_episodes"

    # ROOT = Path("/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs_1112/grpo_olmo_1B_gsm8k_baseline_800000_episodes")
    # REPO_NAME = "20251112-grpo_olmo_1B_gsm8k_baseline_800000_episodes"

    # ROOT = Path("/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs_1112/grpo_olmo_1B_with_classmate_llama_800000_episodes")
    # REPO_NAME = "20251112-grpo_olmo_1B_with_classmate_llama_800000_episodes"

    ROOT = Path("/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs_1112/grpo_olmo_1B_with_classmate_llama_reward_weight_1_400000_episodes")
    REPO_NAME = "20251112-grpo_olmo_1B_with_classmate_llama_reward_weight_1_400000_episodes"

    PRIVATE = False                                # set False to make public

    api = HfApi()
    repo_id = f"{USER_OR_ORG}/{REPO_NAME}"

    # Create the single repo if missing
    try:
        api.repo_info(repo_id, repo_type="model")
        print(f"[info] Repo exists: {repo_id}")
    except Exception:
        print(f"[create] {repo_id} (private={PRIVATE})")
        create_repo(repo_id, private=PRIVATE, repo_type="model", exist_ok=True)

    # Discover candidate subdirs (e.g., global_step_114, global_step_133, ...)
    sub_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
    skip_dir = [".hydra"]
    sub_dirs = [p for p in sub_dirs if p.name not in skip_dir]
    sub_dirs.sort(key=lambda p: int(p.name.split('_')[-1]))
    # total_cnt = len(sub_dirs)
    # total_cnt = int(total_cnt * 3 / 4)  # only upload last 3/4 checkpoints
    # sub_dirs = sub_dirs[total_cnt:]  # adjust if you want to limit number of uploads

    for step_dir in sub_dirs:
        hf_dir = step_dir / "actor" / "huggingface"
        if not hf_dir.exists():
            print(f"[skip] {step_dir.name}: {hf_dir} not found")
            continue

        # Use step_dir.name as the revision/branch name
        revision = step_dir.name
        print(f"\n=== Uploading {step_dir.name} -> {repo_id} (revision: {revision}) ===")

        # Load tokenizer (if present)
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_dir, use_fast=True)
        except Exception as e:
            print(f"[warn] tokenizer load failed for {step_dir.name}: {e}")

        # Load model (CPU; avoid GPU OOM on many checkpoints)
        try:
            model = AutoModelForCausalLM.from_pretrained(hf_dir)
        except Exception as e:
            print(f"[error] model load failed for {step_dir.name}: {e}")
            continue

        # Push model weights/config to specific revision
        model.push_to_hub(
            repo_id,
            revision=revision,
            create_pr=False,  # Create a new branch instead of PR
            use_temp_dir=True,
            commit_message=f"Upload checkpoint {step_dir.name}",
            private=PRIVATE,
        )
        print("[ok] model pushed")

        # Push tokenizer (if we loaded it) to the same revision
        if tokenizer is not None:
            try:
                tokenizer.push_to_hub(
                    repo_id,
                    revision=revision,
                    create_pr=False,
                    use_temp_dir=True,
                    commit_message=f"Add tokenizer for {step_dir.name}",
                    private=PRIVATE,
                )
                print("[ok] tokenizer pushed")
            except Exception as e:
                print(f"[warn] tokenizer push failed: {e}")

        # Push generation config if present to the same revision
        gen_cfg = None
        try:
            gen_cfg = GenerationConfig.from_pretrained(hf_dir)
        except Exception:
            gen_cfg = None
        if gen_cfg is not None:
            try:
                gen_cfg.push_to_hub(
                    repo_id,
                    revision=revision,
                    create_pr=False,
                    use_temp_dir=True,
                    commit_message=f"Add generation config for {step_dir.name}",
                    private=PRIVATE,
                )
                print("[ok] generation_config pushed")
            except Exception as e:
                print(f"[warn] generation_config push failed: {e}")

        print(f"[done] {repo_id} @ {revision}")
        
    print(f"\n=== Upload Complete ===")
    print(f"All checkpoints uploaded to: {repo_id}")
    print(f"To load a specific checkpoint, use:")
    print(f'model = AutoModelForCausalLM.from_pretrained("{repo_id}", revision="global_step_XXX")')
    print(f"Available revisions:")
    for step_dir in sub_dirs:
        if (step_dir / "actor" / "huggingface").exists():
            print(f"  - {step_dir.name}")


def download_huggingface_ckpt():
    from transformers import AutoModelForCausalLM

    model_name_path = "LorenaYannnnn/classmate-cot-baseline"
    all_revision_steps = [f"global_step_{step_idx * 19}" for step_idx in range(1, 20)] + ["global_step_364"]
    output_dir = "/local/data/lorena/grpo_olmo_1B_gsm8k_baseline"
    for revision in all_revision_steps:
        save_path = f"{output_dir}/{revision}/actor/huggingface"
        if os.path.exists(save_path):
            print(f"Skipped revision {revision}, already exists at {save_path}.")
            continue
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_path, revision=revision)
            model.save_pretrained(save_path)
            print(f"Saved checkpoint {revision} to {save_path}")
        except:
            print(f"Skipped revision {revision}, could not be loaded.")


def test():
    import ast

    def execute_code(solution_code, test_case_code, import_prompt, entry_point, check_correctness=False):
        """
        Execute the solution code + test code to verify executability.
        Optionally run assertion tests for correctness.
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


    import_prompt = "from typing import List"

    entry_point = "Solution().twoSum"

    solution_code = """class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        d = {}
        for i, x in enumerate(nums):
            if (target - x) in d:
                return [d[target - x], i]
            d[x] = i
    """

    test_case_code = """def check(candidate):
    assert candidate(nums=[3, 3], target=6) == [0, 1]
    assert candidate(nums=[-1, -2, -3, -4], target=-8) is None
    assert candidate(nums=[1000000000, 1000000000], target=2000000000) == [0, 1]
    """

    # Test both modes
    result_correctness = execute_code(solution_code, test_case_code,
                                      import_prompt, entry_point,
                                      check_correctness=True)

    result_executable = execute_code(solution_code, test_case_code,
                                     import_prompt, entry_point,
                                     check_correctness=False)

    print("Executability test:", result_executable)
    print("Correctness test:", result_correctness)

    input_prompt = """### Question:
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

Example 1:

Input: nums = [2,7,11,15], target = 9
Output: [0,1]
Explanation: Because nums[0] + nums[1] == 9, we return [0, 1].

Example 2:

Input: nums = [3,2,4], target = 6
Output: [1,2]

Example 3:

Input: nums = [3,3], target = 6
Output: [0,1]

Constraints:

2 <= nums.length <= 104
-109 <= nums[i] <= 109
-109 <= target <= 109
Only one valid answer exists.

### Format: Use the following starter code to write the solution and one or more assert statements to test the solution.
````python
# 1. Solution:
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

# 2. Test case:
def check(candidate):
    assert candidate(

```

### Answer: Return only the code and the check function.
```python
# 1. Solution:
"""

    model_name = "allenai/OLMo-2-0425-1B-DPO"
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    tokenized_input = tokenizer(input_prompt, return_tensors="pt")
    tokenized_input = {k: v.to("cuda") for k, v in tokenized_input.items()}
    generation_output = model.generate(**tokenized_input, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
    generated_text = tokenizer.decode(generation_output[0][len(tokenized_input["input_ids"][0]):], skip_special_tokens=True)
    print(generated_text)



if __name__ == "__main__":
    # upload_ckpts_to_huggingface()
    # download_huggingface_ckpt()
    test()

#CUDA_VISIBLE_DEVICES=2 python test.py
