from together import Together

vllm_configs = {
    "enable_sleep_mode": True,
    "enforce_eager": False,
    "disable_custom_all_reduce": True,
    "skip_tokenizer_init": False,
    "disable_log_stats": True,
    "max_num_batched_tokens": 8192,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "trust_remote_code": False
}

def load_model(configs):
    """main function for loading the model_module"""
    # if "open-instruct/outputs" in configs.model_args.model_name_or_path and hasattr(configs.model_args, "main_model_step_idx"):
    #     model = AutoModelForCausalLM.from_pretrained(
    #         f"{configs.model_args.model_name_or_path}/step_{configs.model_args.main_model_step_idx}",
    #         device_map="auto",
    #         dtype="auto"
    #     )
    #     # model = LLM(
    #     #     model=f"{configs.model_args.model_name_or_path}/step_{configs.model_args.main_model_step_idx}",
    #     #     dtype="auto"
    #     # )
    if hasattr(configs.model_args, "main_model_step_idx"):
        is_base = configs.model_args.main_model_step_idx == "base"
        if "LorenaYannnnn/" in configs.model_args.model_name_or_path:
            print("Load from huggingface")
            model_configs = {
                "pretrained_model_name_or_path": configs.model_args.base_model_name_or_path if is_base else configs.model_args.model_name_or_path,
                "revision": "main" if is_base else f"global_step_{configs.model_args.main_model_step_idx}",
            }
        elif "/local/data/lorena/" in configs.model_args.model_name_or_path or "/proj/interaction/interaction-filer/lorena/" in configs.model_args.model_name_or_path:
            print("Load from local path")
            model_configs = {
                "pretrained_model_name_or_path": configs.model_args.base_model_name_or_path if is_base else f"{configs.model_args.model_name_or_path}/global_step_{configs.model_args.main_model_step_idx}/actor/huggingface",
            }
        else:
            raise NotImplementedError(f"Model {configs.model_args.model_name_or_path} with main_model_step_idx not supported yet.")
            # model = LLM(
            #     model=configs.model_args.base_model_name_or_path if is_base else configs.model_args.model_name_or_path,
            #     revision="main" if is_base else f"global_step_{configs.model_args.main_model_step_idx}",
            #     dtype="auto"
            # )
    else:
        print(f"Loading model from {configs.model_args.model_name_or_path}, step idx: {getattr(configs.model_args, 'main_model_step_idx', 'main')}")
        model_configs = {
            "pretrained_model_name_or_path": configs.model_args.model_name_or_path,
            "revision": f"step_{configs.model_args.main_model_step_idx}" if hasattr(configs.model_args, "main_model_step_idx") and configs.model_args.main_model_step_idx != "main" else "main",
        }

    if configs.model_args.inference_backend == "vllm":
        from vllm import LLM
        model_configs["model"] = model_configs.pop("pretrained_model_name_or_path")
        model_configs["gpu_memory_utilization"] = configs.model_args.vllm_gpu_memory_utilization
        model_configs["dtype"] = configs.model_args.dtype if hasattr(configs.model_args, "dtype") else "auto"
        model_configs.update(vllm_configs)
        model = LLM(**model_configs)
    elif configs.model_args.inference_backend == "Together":
        model = Together()
        model.model_name = configs.model_args.model_name_or_path + ("-Turbo" if "Turbo" not in configs.model_args.model_name_or_path else "")
    else:
        raise NotImplementedError(f"Inference backend {configs.model_args.inference_backend} not supported yet.")
        # from transformers import AutoModelForCausalLM
        # model_configs["device_map"] = "auto"
        # model = AutoModelForCausalLM.from_pretrained(**model_configs)

    # Set up LoRA
    # if configs.training_args.resume_from_checkpoint:
    #     with open(os.path.join(configs.training_args.resume_from_checkpoint, "adapter_config.json")) as f:
    #         # Convert to dict
    #         config_params = json.load(f)
    #         config = LoraConfig(**config_params)
    #         if configs.training_args.do_train or (configs.training_args.do_predict and configs.training_args.save_grads):
    #             config.inference_mode = False
    # else:
    #     config = LoraConfig(
    #         r=configs.model_args.lora_r,
    #         lora_alpha=configs.model_args.lora_alpha,
    #         lora_dropout=configs.model_args.lora_dropout,
    #         target_modules=list(configs.model_args.lora_target_modules),
    #         bias="none",
    #         task_type="CAUSAL_LM"
    #     )
    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    return model
