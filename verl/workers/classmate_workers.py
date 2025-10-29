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
from dataclasses import dataclass
from typing import Dict, Any
import logging
import os

import numpy as np
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
# from tqdm import tqdm

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
    max_prompt_len: int
    classmate_batch_size: int
    classmate_free_cache_engine: bool
    classmate_use_vllm: bool

    generation_config: Dict[str, Any]
    """
    max_new_tokens: 128
    do_sample: True
    temperature: 0.7
    top_p: 0.9
    num_return_sequences: 1
    """

    vllm_config: Dict[str, Any]
    """
    gpu_memory_utilization: 0.9
    tensor_parallel_size: 1
    """


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
        config.vllm_config.max_model_len = config.max_prompt_len + config.generation_config.max_new_tokens
        self.vllm_config = config.vllm_config

        # Initialize model and tokenizer as None - will be loaded in init_model()
        self.tokenizer = None
        self.llm = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Initialize the single classmate model for this worker.
        
        Follows the same pattern as ActorRolloutRef workers:
        1. Load the model
        2. Immediately offload it to free GPU memory
        3. Model will be loaded on-demand during generation
        """
        print(f"Initializing classmate model {self.model_index}: {self.model_name_or_path}")

        try:
            # Load tokenizer (always needed)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            # Ensure valid padding token and left padding for causal LM
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            # Left-pad so newest tokens align right for causal models
            self.tokenizer.padding_side = "left"

            if self.use_vllm:
                raise NotImplementedError("vLLM support is currently disabled for classmate workers.")
                from vllm import LLM
                import torch
                self.llm = LLM(
                    model=self.model_name_or_path,
                    **self.vllm_config,
                    trust_remote_code=True,
                    dtype="auto",
                )
                log_gpu_memory_usage(f"After loading classmate model {self.model_index}", logger=logger)
                print(f"Successfully loaded classmate model {self.model_index} with vLLM: {self.model_name_or_path}")

                # Immediately offload the model to free GPU memory (following ActorRolloutRef pattern)
                if self.free_cache_engine:
                    self._offload_model()
                    log_gpu_memory_usage(f"After offloading classmate model {self.model_index}", logger=logger)
                    print(f"Offloaded classmate model {self.model_index} to free GPU memory")
            else:
                # Fall back to huggingface
                from transformers import AutoModelForCausalLM
                # Load on CPU first; we'll move to the single visible GPU in _onload_model.
                # Avoid device_map="auto" here to prevent unintended tensor-parallel sharding across GPUs
                # inside each Ray worker, which can cause contention and incoherent outputs.
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
        """Load vLLM engine to GPU memory (wake up).
        
        Follows the same pattern as vLLM rollout's resume() method.
        Simply calls wake_up() if available - vLLM handles the internal details.
        """
        if self.use_vllm:
            if self.llm is None:
                raise RuntimeError(f"vLLM engine for classmate model {self.model_index} is not initialized")

            if not self.free_cache_engine:
                return  # No need to wake up if sleep mode is disabled

            try:
                # Wake up the vLLM engine (load weights and KV cache back to GPU)
                if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'wake_up'):
                    # For LLM with llm_engine attribute
                    self.llm.llm_engine.wake_up()
                elif hasattr(self.llm, 'wake_up'):
                    # For regular LLM class
                    self.llm.wake_up()

                log_gpu_memory_usage(f"After onloading classmate model {self.model_index}", logger=logger)
            except Exception as e:
                logger.warning(f"Failed to wake up classmate model {self.model_index}: {e}")
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
        """Offload vLLM engine from GPU memory (sleep).
        
        Follows the same pattern as vLLM rollout's release() method.
        Uses aggressive_empty_cache for cleanup like FSDP workers.
        """
        if self.use_vllm:
            if self.llm is None:
                return

            if not self.free_cache_engine:
                return  # No need to sleep if sleep mode is disabled

            try:
                # Put the vLLM engine to sleep (offload weights and KV cache from GPU)
                if hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'sleep'):
                    # For LLM with llm_engine attribute
                    self.llm.llm_engine.sleep()
                elif hasattr(self.llm, 'sleep'):
                    # For regular LLM class
                    self.llm.sleep()

                # Aggressive cache cleanup (following FSDP workers pattern)
                aggressive_empty_cache(force_sync=True)
                log_gpu_memory_usage(f"After offloading classmate model {self.model_index}", logger=logger)
            except Exception as e:
                logger.warning(f"Failed to sleep classmate model {self.model_index}: {e}")
        else:
            if self.llm is None:
                return
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
        #     "prompts": all_prompts,
        #     "prompt_plus_actor_responses": all_prompt_plus_actor_responses
        # }
        classmate_outputs = []
        try:
            # ONLOAD: Load model to GPU before generation (following ActorRolloutRef pattern)
            self._onload_model()
            log_gpu_memory_usage(f"After onloading classmate model {self.model_index} for generation", logger=logger)
            
            prompt_plus_actor_responses = data.non_tensor_batch["prompt_plus_actor_responses"]
            prompts = data.non_tensor_batch["prompts"]
            
            # Convert numpy array to list if needed
            if isinstance(prompt_plus_actor_responses, np.ndarray):
                prompt_plus_actor_responses = prompt_plus_actor_responses.tolist()
            # if isinstance(prompts, np.ndarray):
            #     prompts = prompts.tolist()
            
            num_return_sequences = self.generation_config.num_return_sequences
            assert num_return_sequences == 1, "Classmate worker currently only supports num_return_sequences=1"

            print(f"Start sampling from classmate model {self.model_index} for {len(prompt_plus_actor_responses)} inputs")

            if self.use_vllm:
                raise NotImplementedError("vLLM support is currently disabled for classmate workers.")
                from vllm import SamplingParams

                # Convert generation_config to vLLM SamplingParams
                sampling_params = SamplingParams(
                    n=num_return_sequences,
                    **self.generation_config,
                    skip_special_tokens=True,
                )

                # Process in batches to manage memory
                total_samples = len(prompt_plus_actor_responses)
                for batch_start in range(0, total_samples, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, total_samples)
                    batch_prompts = prompt_plus_actor_responses[batch_start:batch_end]
                    
                    # Generate with vLLM for this batch
                    batch_outputs = self.llm.generate(batch_prompts, sampling_params)
                    
                    # Extract outputs (vLLM returns RequestOutput objects)
                    for output in batch_outputs:
                        sample_outputs = [completion.text for completion in output.outputs]
                        classmate_outputs.append(sample_outputs)
            else:
                # Fall back to huggingface generation
                import torch

                # Data collator to left-pad using tokenizer's padding_side
                collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt")

                # TODO: need to fix this for supporting num_return_sequences > 1
                # Process in batches to manage memory
                total_samples = len(prompt_plus_actor_responses)
                # for batch_start in tqdm(range(0, total_samples, self.batch_size), desc=f"Classmate Model {self.model_index} Generating"):
                for batch_start in range(0, total_samples, self.batch_size):
                    batch_end = min(batch_start + self.batch_size, total_samples)
                    batch_prompts_with_responses = prompt_plus_actor_responses[batch_start:batch_end]
                    # batch_prompts_only = prompts[batch_start:batch_end]
                    
                    # Tokenize the full prompt+actor_response for generation (no padding here)
                    request_features = [
                        self.tokenizer(t, padding=False, return_attention_mask=True)
                        for t in batch_prompts_with_responses
                    ]
                    encoded_batch = collator(request_features)
                    # Move batch tensors to the same device as the model parameters
                    import torch
                    model_device = next(self.llm.parameters()).device
                    encoded_batch = {k: (v.to(model_device) if isinstance(v, torch.Tensor) else v) for k, v in encoded_batch.items()}

                    # Generate outputs for this batch
                    with torch.no_grad():
                        generated_outputs = self.llm.generate(
                            **encoded_batch,
                            **self.generation_config,
                        )

                    # print("🐛🐛🐛 generated_outputs", generated_outputs)
                    # print("🐛🐛🐛 generated_outputs shape", generated_outputs.shape)   # (bsz, seq_len)
                    # print("🐛🐛🐛 encoded_batch", encoded_batch.keys())   # (bsz, seq_len)
                    # print("🐛🐛🐛 encoded_batch", encoded_batch)   # (bsz, seq_len)
                    # print("🐛🐛🐛 encoded_batch attention mask", encoded_batch["attention_mask"].shape)   # (bsz, seq_len)

                    # encoded_prompts_only = self.tokenizer(
                    #     batch_prompts_only,
                    #     padding=False,
                    #     return_attention_mask=False,
                    # )
                    current_batch_size = len(batch_prompts_with_responses)
                    # Process each item in the current batch
                    for batch_idx in range(current_batch_size):
                        # Extract only classmate continuation
                        # With left padding, the input length equals padded length (max_len),
                        # which is the prefix of the generated sequence returned by HF generate.
                        # So slicing by that length yields only the newly generated tokens.
                        input_len = encoded_batch["input_ids"].shape[1]
                        generated_tokens = generated_outputs[batch_idx][input_len:]
                        decoded_output = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                        # Extract out actor + classmate CoT
                        # sample_outputs = []
                        # #
                        # # With left padding: padding_len = max_len - actual_len
                        # max_len_with_responses = encoded_batch["attention_mask"][batch_idx].size(0)
                        # actual_len = int(encoded_batch["attention_mask"][batch_idx].sum().item())
                        # padding_len = max_len_with_responses - actual_len
                        # # Prompt-only length computed from non-padded tokenized prompt
                        # prompt_only_len = len(encoded_prompts_only["input_ids"][batch_idx])
                        # cot_start_idx = padding_len + prompt_only_len
                        # # for seq_idx in range(num_return_sequences):
                        # # Extract and decode CoT tokens (actor response + classmate continuation)
                        # cot_tokens = generated_outputs[batch_idx][cot_start_idx:]
                        # #
                        # # decoded_output = self.tokenizer.decode(cot_tokens, skip_special_tokens=True)
                        # # sample_outputs.append(decoded_output)

                        # print(f"""
                        # 🐛Full input output: {self.tokenizer.decode(generated_outputs[batch_idx], skip_special_tokens=True)}
                        # 🐛Input to classmate: {self.tokenizer.decode(encoded_batch["input_ids"][batch_idx], skip_special_tokens=True)}
                        # 🐛classmate continuation: {decoded_output}
                        # """)
                        classmate_outputs.append(decoded_output)

        except Exception as e:
            print(f"❌ Error generating with classmate model {self.model_index}: {e}")
            import traceback
            traceback.print_exc()
            # Return empty outputs matching expected shape
            bsz = len(data.non_tensor_batch.prompt_plus_actor_responses)
            num_return_sequences = self.generation_config.num_return_sequences
            classmate_outputs = [[""] * num_return_sequences for _ in range(bsz)]
        
        finally:
            # OFFLOAD: Offload model from GPU after generation (following ActorRolloutRef pattern)
            self._offload_model()
            log_gpu_memory_usage(f"After offloading classmate model {self.model_index} post-generation", logger=logger)

        # TODO need to fix this to support num_return_sequences > 1
        classmate_outputs = [[output] for output in classmate_outputs]  # Wrap each output in a list

        result_non_tensor_batch = {
            "classmate_output": np.array(classmate_outputs),      # Should have shape (bsz, num_return_sequences)
        }

        # print(f"🐛 From classmateWorker {self.model_index} classmate_output", result_non_tensor_batch["classmate_output"])

        return DataProto(batch=None, non_tensor_batch=result_non_tensor_batch)


def create_classmate_workers(config):
    """Create individual Ray remote workers for each classmate model.

    Args:
        config: Configuration containing classmate model paths and generation config

    Returns:
        List of Ray remote worker references, one per classmate model
    """
    classmate_config = config.get("classmate_cot_reward_configs", {})
    model_paths = classmate_config.get("classmate_model_name_or_path_list", [])

    # Get generation configuration
    generation_config = config.get("classmate_generation_configs", {})

    assert model_paths, "No classmate models configured"

    workers = []
    for i, model_path in enumerate(model_paths):
        # Create a Ray remote worker for each classmate model
        worker_cls = ray.remote(ClassmateWorker)
        worker = worker_cls.remote(
            model_name_or_path=model_path,
            generation_config=generation_config,
            model_index=i
        )
        workers.append(worker)
        print(f"Created classmate worker {i} for model: {model_path}")

    return workers