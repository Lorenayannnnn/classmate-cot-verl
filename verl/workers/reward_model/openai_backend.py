"""OpenAI-compatible judge/monitor backend."""

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from verl.workers.reward_model.judge_backend import BaseBackend


class OpenAICompatibleBackend(BaseBackend):
    """BaseBackend wrapping an OpenAI-compatible API, with parallel requests via ThreadPoolExecutor."""

    def __init__(self, model_name: str, api_key: str, base_url: str | None = None, max_retries: int = 5):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name
        self._max_retries = max_retries
        self._executor = ThreadPoolExecutor()

    def prepare_input(self, verifier, question: str, output: str) -> str:
        return verifier.format_llm_judge_prompt(question, output)

    def _call_one(self, prompt: str, icl_pairs: list[tuple[str, str]] | None = None) -> str:
        messages = []
        if icl_pairs:
            for user_msg, assistant_msg in icl_pairs:
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": prompt})
        for attempt in range(self._max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[retry {attempt + 1}/{self._max_retries}] Error calling {self._model_name}: {e}")
        return ""

    def run_batch(self, inputs: list, icl_pairs: list[tuple[str, str]] | None = None) -> list:
        futures = {i: self._executor.submit(self._call_one, p, icl_pairs) for i, p in enumerate(inputs) if p is not None}
        results = [None] * len(inputs)
        for i, fut in futures.items():
            results[i] = fut.result()
        return results

    def close(self):
        self._executor.shutdown(wait=False)
