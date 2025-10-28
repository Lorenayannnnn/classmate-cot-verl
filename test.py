from transformers import AutoModelForCausalLM


def upload_ckpts_to_huggingface():
    from pathlib import Path
    from huggingface_hub import HfApi, create_repo
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    ROOT = Path("outputs/classmate_cot_w_verl/grpo_olmo_1B_gsm8k_baseline")
    USER_OR_ORG = "LorenaYannnnn"                 # <-- change if needed
    REPO_NAME = "classmate-cot-baseline"  # single repo name
    PRIVATE = True                                # set False to make public

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
    sub_dirs.sort(key=lambda p: int(p.name.split('_')[-1]))
    total_cnt = len(sub_dirs)
    total_cnt = int(total_cnt * 3 / 4)  # only upload last 3/4 checkpoints
    sub_dirs = sub_dirs[total_cnt:]  # adjust if you want to limit number of uploads

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


if __name__ == "__main__":
    upload_ckpts_to_huggingface()