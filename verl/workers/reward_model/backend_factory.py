"""Factory for constructing BaseBackend instances from Hydra reward_model config."""

from verl.workers.reward_model.judge_backend import BaseBackend

def build_eval_llm_judge_backend(reward_model_cfg) -> BaseBackend:
    """
    Build a BaseBackend for LLM judge from the reward_model section of a Hydra config.
    This is used for _validate only, in case when data_source is general_reward and we want to use a different model for scoring different behaviors of the output.
    Returns None when eval_llm_judge_backend_type or eval_llm_judge_model_name are not set
    (e.g. for non-general_reward datasets that don't use this backend).
    """
    backend_type = reward_model_cfg.get("eval_llm_judge_backend_type")
    model_name = reward_model_cfg.get("eval_llm_judge_model_name")
    api_key = reward_model_cfg.get("api_key")
    if not backend_type or not model_name:
        return None

    return _build_backend(
        backend_type=backend_type,
        model_name=model_name,
        cfg=reward_model_cfg,
        prefix="llm_judge",
        api_key=api_key,
    )


def build_llm_judge_backend(reward_model_cfg) -> BaseBackend:
    """Build a BaseBackend for LLM judge from the reward_model section of a Hydra config.

    Config fields read
    ------------------
    llm_judge_backend_type : str
        One of "tinker" (default), "vllm_generative", "hf_scoring".
    llm_judge_model_name : str
        Model name or path.  Required for all backend types.

    tinker-specific (no extra fields needed beyond llm_judge_model_name).

    vllm_generative-specific:
        llm_judge_max_model_len        : int   (default 8192)
        llm_judge_gpu_memory_utilization: float (default 0.8)
        llm_judge_generation_config    : dict  (e.g. {temperature: 1.0, max_tokens: 1024})

    hf_scoring-specific:
        (no extra fields needed beyond llm_judge_model_name)
    """
    backend_type = reward_model_cfg.get("llm_judge_backend_type")
    assert backend_type, "llm_judge_backend_type must be set"
    model_name = reward_model_cfg.get("llm_judge_model_name")
    assert model_name, "reward_model.llm_judge_model_name must be set."
    api_key = reward_model_cfg.get("api_key")

    return _build_backend(
        backend_type=backend_type,
        model_name=model_name,
        cfg=reward_model_cfg,
        prefix="llm_judge",
        api_key=api_key
    )


def build_monitor_backend(reward_model_cfg) -> BaseBackend:
    """Build a BaseBackend for monitor from the reward_model section of a Hydra config.

    Config fields read
    ------------------
    monitor_backend_type : str
        One of "tinker" (default), "vllm_generative".
    monitor_model_name : str
        Model name or path.  Required for all backend types.

    tinker-specific (no extra fields needed beyond monitor_model_name).

    vllm_generative-specific:
        monitor_max_model_len        : int   (default 8192)
        monitor_gpu_memory_utilization: float (default 0.8)
        monitor_generation_config    : dict  (e.g. {temperature: 0.0, max_tokens: 1024})
    """
    backend_type = reward_model_cfg.get("monitor_backend_type")
    assert backend_type, "monitor_backend_type must be set"
    model_name = reward_model_cfg.get("monitor_model_name")
    assert model_name, "reward_model.monitor_model_name must be set."
    api_key = reward_model_cfg.get("api_key")

    return _build_backend(
        backend_type=backend_type,
        model_name=model_name,
        cfg=reward_model_cfg,
        prefix="monitor",
        api_key=api_key
    )

def _vllm_generation_config(model_name: str) -> dict:
    """Return model-specific vLLM sampling params."""
    if model_name == "HuggingFaceTB/SmolLM2-360M-Instruct":
        # https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct
        return {"temperature": 0.2, "top_p": 0.9, "max_tokens": 1024}
    # if model_name == "Qwen/Qwen2.5-0.5B-Instruct":
    #     # https://qwen.ai/blog?id=qwen2.5
    #     return {"temperature": 0.7, "top_p": 0.8, "repetition_penalty": 1.05, "max_tokens": 1024}
    else:
        raise ValueError(f"Make sure to have the sampling param for model {model_name}")


def _build_backend(backend_type: str, model_name: str, prefix: str, cfg=None, api_key: str | None = None) -> BaseBackend:
    """Shared construction logic for judge and monitor backends.

    Args:
        backend_type: One of "tinker", "vllm_generative", "hf_scoring", "openai".
        model_name: HuggingFace model id or local path.
        cfg: Full reward_model config dict (OmegaConf or dict).
        prefix: Config key prefix ("llm_judge" or "monitor") for backend-specific params.
        api_key: API key for "openai" backend.
    """
    if backend_type == "tinker":
        from verl.utils.reward_score.monitor import create_llm_judge
        from verl.workers.reward_model.tinker_backend import TinkerJudgeBackend
        return TinkerJudgeBackend(create_llm_judge(judge_model_name=model_name))

    elif backend_type == "vllm":
        from verl.workers.reward_model.vllm_generative_backend import VLLMGenerativeJudgeBackend
        # return VLLMGenerativeJudgeBackend(
        #     model_name_or_path=model_name,
        #     max_model_len=cfg.get(f"{prefix}_max_model_len", 8192),
        #     generation_config=dict(cfg.get(f"{prefix}_generation_config", {"temperature": 0.0, "max_tokens": 1024})),
        #     gpu_memory_utilization=cfg.get(f"{prefix}_gpu_memory_utilization", 0.8),
        # )
        return VLLMGenerativeJudgeBackend(
            model_name_or_path=model_name,
            generation_config=_vllm_generation_config(model_name),
            gpu_memory_utilization=0.9,
        )

    elif backend_type == "hf_scoring":
        if prefix == "monitor":
            raise ValueError("hf_scoring is not supported for monitor backends (monitor requires generative output).")
        # Return a placeholder backend; the Ray ScoringJudgeWorker is wired in after init_workers().
        from verl.workers.reward_model.scoring_judge_worker_backend import ScoringJudgeWorkerBackend
        return ScoringJudgeWorkerBackend()

    elif backend_type == "openai":
        from verl.workers.reward_model.openai_backend import OpenAICompatibleBackend
        return OpenAICompatibleBackend(model_name=model_name, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown backend_type: {backend_type!r}. "
            "Choose from 'tinker', 'vllm_generative', 'hf_scoring', 'openai'."
        )
