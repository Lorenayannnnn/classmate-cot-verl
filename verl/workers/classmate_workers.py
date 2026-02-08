# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Any
import logging
import os
import time

import numpy as np
# from together import Together
from transformers import AutoTokenizer
import torch
from tensordict import TensorDict
import requests
from vllm import LLM, SamplingParams, TokensPrompt

import ray
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch, Execute
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import log_gpu_memory_usage

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ClassmateWorkerConfig:
    model_name_or_path: str
    model_index: int
    # max_new_tokens: int
    max_tokens: int
    classmate_batch_size: int
    classmate_free_cache_engine: bool
    classmate_use_vllm: bool
    enable_thinking: bool

    generation_config: Dict[str, Any]
    """
    max_new_tokens: 128
    do_sample: True
    temperature: 0.7
    top_p: 0.9
    seed: 42
    """

    vllm_config: Dict[str, Any]
    """
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 1
    """

    # API server configuration for vLLM remote serving
    # Deprecated. Now using TogetherAI
    # api_host: str = None
    # api_port: int = None


class ClassmateWorker(Worker):
    """Worker for a single classmate model's chain-of-thought continuation sampling.

    Each worker is completely self-contained:
    - Has its own model and tokenizer
    - Handles input preparation (tokenization) using its own tokenizer
    - Performs sampling/generation using its own model
    - Returns processed results
    - Supports vLLM memory management (offload/onload) to free GPU memory when not in use

    This design allows different classmate models to use different tokenizers
    and ensures proper tokenization for each specific model. It also follows the
    same memory management pattern as ActorRolloutRef workers for efficient GPU usage.
    """

    def __init__(self, config: ClassmateWorkerConfig, role: str):
        """Initialize classmate worker with model path and generation config.

        Args:
            config: ClassmateWorkerConfig containing model path, generation config, and vLLM settings
            role: Role name for this worker
        """
        super().__init__()
        self.model_name_or_path = config.model_name_or_path
        self.model_index = config.model_index

        self.batch_size = config.classmate_batch_size
        self.generation_config = config.generation_config

        self.use_vllm = config.classmate_use_vllm
        self.free_cache_engine = config.classmate_free_cache_engine
        self.vllm_config = config.vllm_config

        # Initialize model and tokenizer as None - will be loaded in init_model()
        self.llm = None
        self.special_ids = None

        # API server configuration for remote vLLM serving
        # self.api_base_url = f"http://{config.api_host}:{config.api_port}/v1"
        # self.client = None

        self.classmate_vllm_configs = {
            "model": self.model_name_or_path,
            "gpu_memory_utilization": 0.8,
            "dtype": "auto",
            "max_model_len": config.max_tokens
        }
        self.classmate_vllm = LLM(**self.classmate_vllm_configs, enable_sleep_mode=True)

        self.tokenizer = self.classmate_vllm.get_tokenizer()

        if self.model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
            print("🐛🐛🐛: Applied custom template for DeepSeek-R1-Distill-Qwen-1.5B to include main CoT")
            # change chat template to not omit main CoT
            jinja_fn = "chat_templates/DeepSeek-R1-Distill-Qwen-1.5B.jinja"
            self.tokenizer.chat_template = open(jinja_fn, "r").read()
        self.enable_thinking = config.enable_thinking
        self._offload_model()
        print(f"🚀🚀🚀 Initialized ClassmateWorker {self.model_index} with vLLM engine for model {self.model_name_or_path}")

    def __del__(self):
        """Cleanup when worker is destroyed."""
        # if self.use_vllm:
        del self.classmate_vllm

    def _generate_via_vllm(self, batch_input_prompt_ids: list, is_eval=False):
        not_none_prompts = [p for p in batch_input_prompt_ids if p is not None]
        is_none_idx = [i for i, p in enumerate(batch_input_prompt_ids) if p is None]
        if is_eval:
            # Greedy
            sampling_params = SamplingParams(
                temperature=0.0,
                top_k=-1,
                top_p=1.0,
                max_tokens=self.generation_config["max_tokens"],
                prompt_logprobs=0  # Request prompt log probabilities
            )
        else:
            sampling_params = SamplingParams(**self.generation_config)
        outputs = self.classmate_vllm.generate(not_none_prompts, sampling_params=sampling_params, use_tqdm=False)

        completions = []
        completion_token_len = []
        prompt_logprobs = []
        for i in range(len(batch_input_prompt_ids)):
            if i in is_none_idx:
                completions.append(None)
                completion_token_len.append(0)
                prompt_logprobs.append(None)
            else:
                output = outputs.pop(0)
                completions.append(output.outputs[0].text)
                completion_token_len.append(len(output.outputs[0].token_ids))

                token_prob = [list(tok.values())[0].logprob for tok in output.prompt_logprobs if tok is not None]
                # decoded_token = "".join([list(tok.values())[0].decoded_token for tok in output.prompt_logprobs if tok is not None])
                prompt_logprobs.append(token_prob)

        return completions, completion_token_len, prompt_logprobs

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the single classmate model for this worker.

        For vLLM mode: Start a separate vLLM API server process
        For HuggingFace mode: Load the model locally
        """
        print(f"Initializing classmate model {self.model_index}: {self.model_name_or_path}")


        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path.replace("-Turbo", ""))       # TogetherAI models have -Turbo suffix
        # Ensure valid padding token and left padding for causal LM
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.special_ids = set(self.tokenizer.all_special_ids)

        try:
            if self.use_vllm:
                # Create OpenAI client
                # from openai import OpenAI
                # self.client = OpenAI(
                #     base_url=self.api_base_url,
                #     api_key="EMPTY",
                # )
                # self.client = Together()
                pass

                # Start vLLM API server process instead of loading model locally
                # self._start_vllm_server()
                # print(f"Successfully started vLLM API server for classmate model {self.model_index}: {self.model_name_or_path}")
            else:
                # Fall back to huggingface
                from transformers import AutoModelForCausalLM
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.model_name_or_path,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    device_map=None,
                )
                # Ensure pad token id is set for generation correctness
                try:
                    self.llm.config.pad_token_id = self.tokenizer.pad_token_id
                except Exception:
                    pass
                self.llm.eval()
                print(f"Successfully loaded classmate model {self.model_index}: {self.model_name_or_path}")

        except Exception as e:
            print(f"Failed to load classmate model {self.model_name_or_path}: {e}")
            raise

    def _onload_model(self):
        """Load model to GPU memory.

        For vLLM API mode: No-op since model runs in separate server process
        For HuggingFace mode: Move model to GPU
        """
        if self.use_vllm:
            # No-op for vLLM API mode - model is managed by separate server process
            self.classmate_vllm.wake_up()
            # return
        else:
            if self.llm is None:
                raise RuntimeError(f"Huggingface model for classmate model {self.model_index} is not initialized")
            # Move the whole model to the single visible GPU for this worker
            import torch
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.llm.to(target_device)
            print(f"[Classmate {self.model_index}] Onloaded model to {target_device}; CVD={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
            log_gpu_memory_usage(f"After onloading classmate model {self.model_index}", logger=logger)

    def _offload_model(self):
        """Offload model from GPU memory.

        For vLLM API mode: Destroy vLLM engine to free GPU memory
        For HuggingFace mode: Move model to CPU and clear cache
        """
        if self.use_vllm:
            self.classmate_vllm.sleep(level=1)
        else:
            self.llm.to("cpu")
            aggressive_empty_cache(force_sync=True)
            log_gpu_memory_usage(f"After offloading classmate model {self.model_index}", logger=logger)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO, execute_mode=Execute.ALL, blocking=False)
    def generate_classmate_continuations(self, data: DataProto) -> DataProto:
        """Generate classmate continuations for actor responses using this worker's model and tokenizer.

        Follows the same pattern as ActorRolloutRef workers:
        1. Load model to GPU (onload)
        2. Perform generation
        3. Offload model from GPU
        4. Clean up memory

        Uses vLLM for fast batched inference.
        """
        # data.non_tensor_batch = {
        #     "raw_prompts": np.array(all_prompts),
        #     "main_model_responses": np.array(all_processed_actor_responses),
        #     "keep_rates": np.array(all_keep_rates),
        #     "all_other_prompts": np.array(all_other_prompts) if all_other_prompts is not None else None,
        #     "all_other_prompt_gts": np.array(all_other_prompt_gts) if all_other_prompt_gts is not None else None,
        #     # "prompt_plus_actor_responses": np.array(all_prompt_plus_actor_responses)
        # }

        is_eval = data.meta_info["is_eval"]

        classmate_outputs = []
        classmate_output_lens = []
        classmate_prompts = []
        classmate_prompt_logprobs = []
        try:
            # ONLOAD: Load model to GPU before generation (no-op for vLLM API mode)
            self._onload_model()
            log_gpu_memory_usage(f"After onloading classmate model {self.model_index} for generation", logger=logger)

            # prompt_plus_actor_responses = data.non_tensor_batch["prompt_plus_actor_responses"]
            # Handle padding: only slice if pad_size is a positive integer
            prompts = data.non_tensor_batch["raw_prompts"]
            actor_responses = data.non_tensor_batch["main_model_responses"]
            all_other_prompts = data.non_tensor_batch.get("all_other_prompts", None)

            # Convert numpy array to list if needed
            # if isinstance(prompt_plus_actor_responses, np.ndarray):
            #     prompt_plus_actor_responses = prompt_plus_actor_responses.tolist()
            if isinstance(prompts, np.ndarray):
                prompts = prompts.tolist()
            if isinstance(actor_responses, np.ndarray):
                actor_responses = actor_responses.tolist()

            # Filter out None (padded ones)
            original_length = len(prompts)
            prompts = [p for p in prompts if p is not None]
            actor_responses = [r for r in actor_responses if r is not None]
            all_other_prompts = [p for p in all_other_prompts] if all_other_prompts is not None else None

            assert self.generation_config.get("num_return_sequences", 1) == 1 or self.generation_config.get("n", 1) == 1, "Classmate worker currently only supports num_return_sequences=1"

            print(f"Start sampling from classmate model {self.model_name_or_path} ({self.model_index}) for {len(actor_responses)} inputs")

            # if self.use_vllm:
            # Use vLLM API server for generation
            # print(f"Generating via vLLM API server at {self.api_base_url}")
            # print(f"Generating from {self.model_name_or_path} via TogetherAI API")

            # Process in batches to manage API requests
            total_samples = len(actor_responses)
            for batch_start in range(0, total_samples, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_samples)
                batch_prompts = prompts[batch_start:batch_end]
                batch_actor_responses = actor_responses[batch_start:batch_end]
                batch_input_prompts = []
                batch_input_prompt_ids = []

                batch_other_prompt = None

                if all_other_prompts is not None:
                    batch_other_prompt = all_other_prompts[batch_start:batch_end]

                # Apply classmate's chat template
                for idx, (tmp_prompt, tmp_response) in enumerate(zip(batch_prompts, batch_actor_responses)):
                    add_generation_prompt = False
                    messages = [{"role": "user", "content": tmp_prompt}]

                    if self.enable_thinking:
                        messages.append({"role": "assistant", "content": f"<think>\n{tmp_response}\n</think>"})
                    else:
                        messages.append({"role": "assistant", "content": tmp_response})

                    if batch_other_prompt is not None:      # help other q
                        messages.append({"role": "user", "content": batch_other_prompt[idx]})
                        add_generation_prompt = True

                    token_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt)
                    if self.model_name_or_path == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
                        token_ids = token_ids[:-1]       # remove new line at the end

                    # Remove trailing special tokens
                    end = len(token_ids)
                    while end > 0 and token_ids[end - 1] in self.special_ids:
                        end -= 1
                    token_ids = token_ids[:end]

                    if len(token_ids) + 1 > self.classmate_vllm_configs["max_model_len"]:       # +1 for generation token (i.e. generate 1 more token will exceed max length)
                        print(f"⚠️ Warning: Input for classmate is too long (have length {len(token_ids)})")
                        batch_input_prompts.append(None)
                        batch_input_prompt_ids.append(None)
                    else:
                        batch_input_prompts.append(self.tokenizer.decode(token_ids, skip_special_tokens=False, add_generation_prompt=False))
                        batch_input_prompt_ids.append(TokensPrompt(prompt_token_ids=token_ids))

                # Get batch-wise max token len
                # prompt_attn_mask = self.tokenizer(
                #     batch_input_prompts,
                #     padding=True,
                #     return_tensors="pt",
                # )["attention_mask"]
                # batch_max_token_len = prompt_attn_mask.sum(dim=1).max().item()

                # Generate via API for this batch
                # batch_completions = self._generate_via_remote_api(batch_input_prompts, batch_max_token_len)
                # batch_completions = self._generate_via_remote_api(batch_input_prompts)

                batch_completions, batch_completion_lens, batch_prompt_logprobs = self._generate_via_vllm(
                    batch_input_prompt_ids, is_eval=is_eval
                )
                classmate_prompts.extend(batch_input_prompts)
                # Extend back to with padding
                classmate_outputs.extend(batch_completions)
                classmate_output_lens.extend(batch_completion_lens)
                classmate_prompt_logprobs.extend(batch_prompt_logprobs)

                # print(f"🐛🐛🐛 Classmate completions: {classmate_outputs}")
        except Exception as e:
            print(f"❌ Error generating with classmate model {self.model_index}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # OFFLOAD: Offload model from GPU after generation (following ActorRolloutRef pattern)
            self._offload_model()
            log_gpu_memory_usage(f"After offloading classmate model {self.model_index} post-generation", logger=logger)

        # Pad classmate_prompt_logprobs to self.classmate_vllm_configs["max_model_len"] with None for inputs that were too long or None
        padded_classmate_prompt_logprobs = []
        padded_classmate_prompt_logprobs_mask = []
        pad_val = 0
        for logprobs in classmate_prompt_logprobs:
            if logprobs is None:
                padded_classmate_prompt_logprobs.append([pad_val] * self.classmate_vllm_configs["max_model_len"])
                padded_classmate_prompt_logprobs_mask.append([pad_val] * self.classmate_vllm_configs["max_model_len"])
            else:
                padded_classmate_prompt_logprobs.append(logprobs + [pad_val] * (self.classmate_vllm_configs["max_model_len"] - len(logprobs)))
                padded_classmate_prompt_logprobs_mask.append([1] * len(logprobs) + [pad_val] * (self.classmate_vllm_configs["max_model_len"] - len(logprobs)))

        # TODO need to fix this to support num_return_sequences > 1
        classmate_outputs = [[output] for output in classmate_outputs]  # Wrap each output in a list
        classmate_output_lens = [[length] for length in classmate_output_lens]
        padded_classmate_prompt_logprobs = [[logprobs] for logprobs in padded_classmate_prompt_logprobs]
        padded_classmate_prompt_logprobs_mask = [[mask] for mask in padded_classmate_prompt_logprobs_mask]

        result_non_tensor_batch = {
            "classmate_prompts": np.array(classmate_prompts),    # Should have shape (bsz,)
            "classmate_outputs": np.array(classmate_outputs),      # Should have shape (bsz, num_return_sequences)
            "classmate_output_lens": np.array(classmate_output_lens),  # Should have shape (bsz, num_return_sequences)
            "classmate_prompt_logprobs": np.array(padded_classmate_prompt_logprobs),  # Should have shape (bsz, num_return_sequences, input_seq_len)
            "classmate_prompt_logprobs_mask": np.array(padded_classmate_prompt_logprobs_mask),  # Should have shape (bsz, num_return_sequences, input_seq_len)
        }

        # print(f"🐛 From classmateWorker {self.model_index} classmate_output", result_non_tensor_batch["classmate_outputs"])
        print(f"Finish sampling from classmate model {self.model_name_or_path} ({self.model_index})")

        return DataProto(batch=None, non_tensor_batch=result_non_tensor_batch)
