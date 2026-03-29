"""
Quick smoke test: check whether the Tinker service is reachable and can
return a response for a trivial prompt.

Usage:
    python test_tinker.py
    python test_tinker.py --model <model_name>  # default: uses whatever create_llm_judge defaults to
    python test_tinker.py --base_url http://...  # optional base URL override
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Tinker model name to test")
    parser.add_argument("--base_url", default=None, help="Tinker service base URL override")
    args = parser.parse_args()

    # ── imports ───────────────────────────────────────────────────────────────
    try:
        from verl.utils.reward_score.monitor import create_llm_judge, send_prompt_to_tinker
    except ImportError as e:
        print(f"[FAIL] Could not import from verl: {e}")
        sys.exit(1)

    # Use a cheap known model if none specified
    model_name = args.model
    if model_name is None:
        # Try to read default from the training config or fall back to a safe default
        try:
            from verl.trainer.config import qwen_classmate_cot_ppo_trainer  # noqa
        except Exception:
            pass
        model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
        print(f"No --model specified, using default: {model_name}")

    # ── create client ─────────────────────────────────────────────────────────
    print(f"Creating Tinker judge client for model: {model_name} ...")
    try:
        config = create_llm_judge(model_name, base_url=args.base_url)
    except Exception as e:
        print(f"[FAIL] create_llm_judge raised: {e}")
        sys.exit(1)
    print("  Client created OK.")

    # ── send a trivial prompt ─────────────────────────────────────────────────
    prompt = "Reply with exactly one word: hello"
    print(f"Sending test prompt: {prompt!r} ...")
    try:
        future = send_prompt_to_tinker(config, prompt)
        result = future.result()
    except Exception as e:
        print(f"[FAIL] send_prompt_to_tinker raised: {e}")
        sys.exit(1)

    # ── extract text ──────────────────────────────────────────────────────────
    try:
        from verl.utils.reward_score.BaseVerifier import BaseVerifier
        text = BaseVerifier._extract_response_text(result, config)
    except Exception as e:
        # Fallback: just show the raw result
        text = str(result)
        print(f"  (Note: _extract_response_text raised {e}; showing raw result)")

    print(f"  Response: {text!r}")
    print("[OK] Tinker is up and responding.")


if __name__ == "__main__":
    main()