import argparse
import base64
import os
import pickle

import torch
import zlib
from datasets import load_dataset
from litellm import acompletion
from transformers import AutoTokenizer, AutoModelForCausalLM


def upload_ckpts_to_huggingface(args):
    from pathlib import Path
    from huggingface_hub import HfApi, create_repo
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig

    USER_OR_ORG = "LorenaYannnnn"  # <-- change if needed
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

    # ROOT = Path("/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes")
    # REPO_NAME = "20251122-grpo_olmo_1B_hendrycks_math_baseline_3298240_episodes"

    # ROOT = Path("/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_allenai/OLMo-2-1124-7B-DPO_RLVR-GSM-MATH-IF-Mixed-Constraints_baseline_237392_episodes")
    # REPO_NAME = "20251210-OLMo-2-1124-7B-DPO_RLVR-GSM-MATH-IF-Mixed-Constraints_baseline_237392_episodes"

    # ROOT = Path("/local/data/lorena/classmate_cot_w_verl/outputs/grpo_allenai/OLMo-2-1124-7B-DPO_RLVR-GSM-MATH-IF-Mixed-Constraints_with_classmate_reward_llama_237400_episodes")
    # REPO_NAME = "20251210-OLMo-2-7B-DPO-Mixed-Constraints_with_classmate_reward_llama_237400_episodes"

    # ROOT = Path("/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-4B-Base_DeepScaleR_baseline_322512_episodes")
    # REPO_NAME = "20251217-Qwen3-4B-Base_DeepScaleR_baseline_322512_episodes"

    # ROOT = Path("/local/data/lorena/classmate_cot_w_verl/outputs/grpo_Qwen/Qwen3-4B-Base_DeepScaleR_w_classmate_llama_322512_episodes")
    # REPO_NAME = "20251217-Qwen3-4B-Base_DeepScaleR_w_classmate_llama_322512_episodes"

    # ROOT = Path("/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-4B-Base_DeepScaleR_w_classmate_llama0.5_322512_episodes")
    # REPO_NAME = "20251217-Qwen3-4B-Base_DeepScaleR_w_classmate_llama0.5_322512_episodes"

    # ROOT = Path("/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-0.6B_GSM_MATH_baseline_357024_episodes")
    # REPO_NAME = "20251224-Qwen3-0.6B_GSM_MATH_baseline_357024_episodes"

    # ROOT = Path("/home/ubuntu/east/classmate-cot-verl/outputs/classmate_cot_w_verl/grpo_Qwen/Qwen3-1.7B_GSM_MATH_baseline_357024_episodes")
    # REPO_NAME = "20251224-Qwen3-1.7B_GSM_MATH_baseline_357024_episodes"

    # ROOT = Path("/local/data/lorena/classmate_cot_w_verl/outputs/grpo_Qwen/Qwen3-1.7B_GSM_MATH_w_1_classmate_llama_357024_episodes")
    # REPO_NAME = "20251224-Qwen3-1.7B_GSM_MATH_w_1_classmate_llama_357024_episodes"

    ROOT = Path(args.root_path)
    REPO_NAME = args.repo_name

    PRIVATE = False  # set False to make public

    api = HfApi()
    repo_id = f"{USER_OR_ORG}/{REPO_NAME}"

    # Create the single repo if missing
    try:
        api.repo_info(repo_id, repo_type="model")
        print(f"[info] Repo exists: {repo_id}")
    except Exception:
        print(f"[create] {repo_id} (private={PRIVATE})")
        create_repo(repo_id, private=PRIVATE, repo_type="model", exist_ok=True)

    # Handle --main_only_path option
    if args.main_only_path:
        main_only_dir = Path(args.main_only_path)
        if not main_only_dir.exists():
            print(f"[error] Path does not exist: {args.main_only_path}")
            return
        
        hf_dir = main_only_dir / "actor" / "huggingface"
        if not hf_dir.exists():
            print(f"[error] HuggingFace directory not found: {hf_dir}")
            return
        
        print(f"\n=== Uploading {main_only_dir.name} -> {repo_id} (revision: main) ===")
        
        # Load tokenizer (if present)
        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_dir)
        except Exception as e:
            print(f"[warn] tokenizer load failed: {e}")
        
        # Load model
        try:
            config = AutoConfig.from_pretrained(hf_dir)
            model = AutoModelForCausalLM.from_pretrained(
                hf_dir,
                config=config,
                torch_dtype=config.dtype,
            )
        except Exception as e:
            print(f"[error] model load failed: {e}")
            return
        
        # Push model to main
        model.push_to_hub(
            repo_id,
            revision="main",
            create_pr=False,
            use_temp_dir=True,
            commit_message=f"Upload checkpoint {main_only_dir.name} to main",
            private=PRIVATE,
        )
        print("[ok] model pushed to main")
        
        # Push tokenizer to main
        if tokenizer is not None:
            try:
                tokenizer.push_to_hub(
                    repo_id,
                    revision="main",
                    create_pr=False,
                    use_temp_dir=True,
                    commit_message=f"Add tokenizer for {main_only_dir.name}",
                    private=PRIVATE,
                )
                print("[ok] tokenizer pushed to main")
            except Exception as e:
                print(f"[warn] tokenizer push failed: {e}")
        
        # Push generation config to main
        gen_cfg = None
        try:
            gen_cfg = GenerationConfig.from_pretrained(hf_dir)
        except Exception:
            gen_cfg = None
        
        if gen_cfg is not None:
            try:
                gen_cfg.push_to_hub(
                    repo_id,
                    revision="main",
                    create_pr=False,
                    use_temp_dir=True,
                    commit_message=f"Add generation config for {main_only_dir.name}",
                    private=PRIVATE,
                )
                print("[ok] generation_config pushed to main")
            except Exception as e:
                print(f"[warn] generation_config push failed: {e}")
        
        print(f"[done] {repo_id} @ main")
        return
    else:
        # Discover candidate subdirs (e.g., global_step_114, global_step_133, ...)
        sub_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
        skip_dir = [".hydra", "wandb"]
        sub_dirs = [p for p in sub_dirs if p.name not in skip_dir]
        sub_dirs.sort(key=lambda p: int(p.name.split('_')[-1]))
        # total_cnt = len(sub_dirs)
        # total_cnt = int(total_cnt * 3 / 4)  # only upload last 3/4 checkpoints
        # sub_dirs = sub_dirs[total_cnt:]  # adjust if you want to limit number of uploads

        upload_dirs = [p for p in sub_dirs if (p / "actor" / "huggingface").exists()]
        last_upload_dir = upload_dirs[-1] if upload_dirs else None

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
                tokenizer = AutoTokenizer.from_pretrained(hf_dir)
            except Exception as e:
                print(f"[warn] tokenizer load failed for {step_dir.name}: {e}")

            # Load model (CPU; avoid GPU OOM on many checkpoints)
            try:
                config = AutoConfig.from_pretrained(hf_dir)
                model = AutoModelForCausalLM.from_pretrained(
                    hf_dir,
                    config=config,
                    torch_dtype=config.dtype,
                )
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

            if last_upload_dir is not None and step_dir == last_upload_dir:
                main_revision = "main"
                print(f"\n=== Also uploading {step_dir.name} -> {repo_id} (revision: {main_revision}) ===")
                model.push_to_hub(
                    repo_id,
                    revision=main_revision,
                    create_pr=False,
                    use_temp_dir=True,
                    commit_message=f"Upload latest checkpoint {step_dir.name} to main",
                    private=PRIVATE,
                )
                print("[ok] model pushed to main")
                if tokenizer is not None:
                    try:
                        tokenizer.push_to_hub(
                            repo_id,
                            revision=main_revision,
                            create_pr=False,
                            use_temp_dir=True,
                            commit_message=f"Add tokenizer for latest {step_dir.name}",
                            private=PRIVATE,
                        )
                        print("[ok] tokenizer pushed to main")
                    except Exception as e:
                        print(f"[warn] tokenizer push to main failed: {e}")
                if gen_cfg is not None:
                    try:
                        gen_cfg.push_to_hub(
                            repo_id,
                            revision=main_revision,
                            create_pr=False,
                            use_temp_dir=True,
                            commit_message=f"Add generation config for latest {step_dir.name}",
                            private=PRIVATE,
                        )
                        print("[ok] generation_config pushed to main")
                    except Exception as e:
                        print(f"[warn] generation_config push to main failed: {e}")

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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--root_path", type=str, required=True, help="Root path to model ckpts."
    )
    arg_parser.add_argument(
        "--repo_name", type=str, required=True, help="Name of the HuggingFace repository."
    )
    arg_parser.add_argument(
        "--main_only_path", type=str, default=None, help="If set, only upload the checkpoint at this path to the 'main' branch."
    )
    upload_ckpts_to_huggingface(arg_parser.parse_args())
    # download_huggingface_ckpt()
    # test()

#python upload_ckpts_to_huggingface.py