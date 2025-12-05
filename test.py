import base64
import os
import pickle

import torch
import zlib
from datasets import load_dataset
from litellm import acompletion
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    # ROOT = Path("/proj/interaction/interaction-filer/lorena/classmate_cot_w_verl/outputs_1112/grpo_olmo_1B_with_classmate_llama_reward_weight_1_400000_episodes")
    # REPO_NAME = "20251112-grpo_olmo_1B_with_classmate_llama_reward_weight_1_400000_episodes"

    ROOT = Path("/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes")
    REPO_NAME = "20251122-grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes"

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


if __name__ == "__main__":
    # upload_ckpts_to_huggingface()
    # download_huggingface_ckpt()
    # test()

    # dataset = load_dataset("allenai/Dolci-Think-RL-7B")
    # # Get unique value of "dataset" column. Note each entry in dataset["train"] is a list
    # all_datasets = [entry["dataset"][0] for entry in dataset["train"]]
    # unique_datasets = set(all_datasets)
    # # get code entries
    #
    # # ['code', 'code_stdio', 'ifeval', 'general-quality', 'general-quality_ref', 'math']
    #
    # # code_entries = [entry for entry in dataset["train"] if entry["dataset"][0] == "code"]
    # general_entries = [entry for entry in dataset["train"] if "general" in entry["dataset"][0].lower()]
    # breakpoint()

    # read in /data/Dolci-Think-RL-7B/train.parquet
    # {'math': 7543, 'code_stdio': 2318, 'code': 3028, 'ifeval': 7453, 'general-quality_ref': 4188, 'general-quality': 970}
    # print number of entries in each subset ['code', 'code_stdio', 'ifeval', 'general-quality', 'general-quality_ref', 'math']
    def count(dataset, split):
        subset_counts = {}
        for entry in dataset:
            dataset_name = entry["data_source"]
            if dataset_name not in subset_counts:
                subset_counts[dataset_name] = 0
            subset_counts[dataset_name] += 1
        print(split, subset_counts)

    dir_name = "/home/lorenayan/data/think-wo_general-Dolci-Think-RL-7B_specified_percentage_general-quality-0_general-quality_ref-0_subset_25500"
    train_dataset = load_dataset("parquet", data_files=f"{dir_name}/train.parquet")["train"]
    eval_dataset = load_dataset("parquet", data_files=f"{dir_name}/eval.parquet")["train"]
    count(train_dataset, "train")
    count(eval_dataset, "eval")
    breakpoint()
    # original_dataset = load_dataset("allenai/Dolci-Think-RL-7B")["train"]
    # Get one entry with dataset == "code_stdio"
    # code_stdio_entry = next(entry for entry in original_dataset if entry["dataset"][0] == "code_stdio" and "eJx9lbtSwzAQRSETHvFXuE" in entry["ground_truth"][0])

    # tests = "eJx1jr0KAjEQhFVOxWAniI0whcVVkr/7exhL4WxOi7tS8AFSxvd1N+HIIboQstl8Mzuv7N3NZ6EuPn/63C1v3WPovTsYGFFASWgpFDS96KAQ3q3uQx+YDf2WDEg/kLZduL2FjjQs3UZEvW8ztw0t8RI28UXiyVvR4U3MU2NRooZJNGdSIUVKZAO91lD1xPik2VhycT4qlnxNWLhrYpkKTRXn1Whiw76phMKMXv++2PT4wxT0Gu2v5w+hDlOO"
    #
    # b64_decoded = base64.b64decode(tests.encode("utf-8"))
    #
    # # Use a streaming decompressor to handle potentially very large test cases
    # # without allocating a massive buffer upfront. This is more memory-efficient.
    # decompressor = zlib.decompressobj()
    # decompressed_chunks = []
    # total_decompressed_size = 0
    #
    # # Process in chunks to avoid holding the entire decompressed data in memory at once.
    # chunk_size = 256 * 1024  # 256KB chunks
    # for i in range(0, len(b64_decoded), chunk_size):
    #     chunk = b64_decoded[i: i + chunk_size]
    #     decompressed_chunk = decompressor.decompress(chunk)
    #     total_decompressed_size += len(decompressed_chunk)
    #     decompressed_chunks.append(decompressed_chunk)
    #
    # decompressed_chunks.append(decompressor.flush())
    #
    # decompressed_data = b"".join(decompressed_chunks)
    # final = pickle.loads(decompressed_data)
    # breakpoint()
    #
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-7B-Think-DPO")

    # model_name_path = "allenai/Olmo-3-7B-Think-DPO"
    # tokenizer = AutoTokenizer.from_pretrained(model_name_path, use_fast=True)
    # messages = [
    #     {"role": "system", "content": "You are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai."},
    #     {"role": "user", "content": "create 12 fortune telling phrases in style of voice phone calling style."}
    # ]
    # inputs = tokenizer.apply_chat_template(messages,
    #     template_name="olmo_thinker",
    #     eos_token=tokenizer.eos_token,
    #     add_generation_prompt=True,
    #     return_tensors="pt"
    # )
    # inputs = {
    #     "input_ids": inputs.cuda(),
    #     "attention_mask": torch.tensor([1] * inputs.shape[-1]).unsqueeze(0).cuda()
    # }
    # model = AutoModelForCausalLM.from_pretrained(model_name_path).to("cuda")
    # outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9)
    # decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # # print(decoded_output)
    # # breakpoint()

    #
    # from litellm import completion
    # import os
    #
    # response = completion(
    #     model="deepinfra/Qwen/Qwen3-32B",
    #     messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
    # )

#CUDA_VISIBLE_DEVICES=3 python test.py
#LAMBDA_API_KEY=secret_lorena_88c093e313d24a0d921ecbe0a5332a8d.IErX5Qxijmx0DpxHrwdUyK2G8s61uFR8 HOSTED_VLLM_API_BASE=https://api.lambda.ai/v1 python test.py
#DEEPINFRA_API_KEY=yafq4uYi5X3T2II8d1aFrBhaYkqy16Ur python test.py
