"""Factory for constructing LLMJudgeBackend instances from Hydra reward_model config."""

from verl.workers.reward_model.judge_backend import LLMJudgeBackend


def build_llm_judge_backend(reward_model_cfg) -> LLMJudgeBackend:
    """Build an LLMJudgeBackend from the reward_model section of a Hydra config.

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
        llm_judge_device               : str   (default "cuda")
    """
    backend_type = reward_model_cfg.get("llm_judge_backend_type")
    assert backend_type, "backend_type must be set"
    model_name = reward_model_cfg.get("llm_judge_model_name")
    assert model_name, "reward_model.llm_judge_model_name must be set."

    if backend_type == "tinker":
        from verl.utils.reward_score.monitor import create_llm_judge
        from verl.workers.reward_model.tinker_backend import TinkerJudgeBackend
        return TinkerJudgeBackend(create_llm_judge(judge_model_name=model_name))
    elif backend_type == "hf_scoring":
        from verl.workers.reward_model.vllm_scoring_backend import HFScoringJudgeBackend
        return HFScoringJudgeBackend(model_name_or_path=model_name)
    elif backend_type == "vllm_generative":
        raise NotImplementedError("vllm_generative not implemented yet.")
        # from verl.workers.reward_model.vllm_generative_backend import VLLMGenerativeJudgeBackend
        # return VLLMGenerativeJudgeBackend(
        #     model_name_or_path=model_name,
        #     max_model_len=reward_model_cfg.get("llm_judge_max_model_len", 8192),
        #     generation_config=dict(reward_model_cfg.get("llm_judge_generation_config", {"temperature": 1.0, "max_tokens": 1024})),
        #     gpu_memory_utilization=reward_model_cfg.get("llm_judge_gpu_memory_utilization", 0.8),
        # )
    else:
        raise ValueError(
            f"Unknown llm_judge_backend_type: {backend_type!r}. "
            "Choose from 'tinker', 'vllm_generative', 'hf_scoring'."
        )
