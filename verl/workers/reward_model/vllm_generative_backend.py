
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from verl.workers.reward_model.judge_backend import BaseBackend


class VLLMGenerativeJudgeBackend(BaseBackend):
    """LLM judge backend using a locally hosted vLLM generative model.

    All non-None prompts are batched into a single vLLM generate() call.
    GPU memory is managed via wake_up() / sleep(level=1) around each call,
    following the same pattern as ClassmateWorker.

    Returns plain text strings so callers can pass the result directly to
    ``verifier.parse_llm_judge_output(text, None)``.
    """

    def __init__(
        self,
        model_name_or_path: str,
        generation_config: dict,
        gpu_memory_utilization: float = 0.8,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model id or local path.
            generation_config: Kwargs forwarded to vLLM SamplingParams
                (e.g. temperature, top_p, max_tokens).
            gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        """
        self.generation_config = generation_config
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._llm = LLM(
            model=model_name_or_path,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="auto",
            enable_sleep_mode=True,
        )
        # Offload immediately after init so the actor/rollout can use the GPU.
        self._llm.sleep(level=1)

    def prepare_input(self, verifier, question: str, output: str) -> str:
        return verifier.format_llm_judge_prompt(question, output)

    def run_batch(
        self,
        prompts: list[str | None],
        icl_pairs: list[tuple[str, str]] | None = None,
    ) -> list[str | None]:
        valid = [(i, p) for i, p in enumerate(prompts) if p is not None]
        if not valid:
            return [None] * len(prompts)

        def _apply_template(prompt: str) -> str:
            messages = []
            if icl_pairs:
                for user_msg, assistant_msg in icl_pairs:
                    messages.append({"role": "user", "content": user_msg})
                    messages.append({"role": "assistant", "content": assistant_msg})
            messages.append({"role": "user", "content": prompt})
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        self._llm.wake_up()
        try:
            sampling_params = SamplingParams(**self.generation_config)
            outputs = self._llm.generate(
                [_apply_template(p) for _, p in valid],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        finally:
            self._llm.sleep(level=1)

        results: list[str | None] = [None] * len(prompts)
        for (i, _), out in zip(valid, outputs):
            results[i] = out.outputs[0].text

        return results

    def close(self):
        del self._llm
