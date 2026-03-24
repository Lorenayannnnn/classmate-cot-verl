import torch
import hydra

from my_eval.src.common_utils import load_config_and_setup_output_dir
from my_eval.src.data_module.load_data import load_data_from_hf, setup_dataloader
from my_eval.src.data_module.preprocessing import preprocess
from my_eval.src.model_module.load_model import load_model
from my_eval.src.train_module.InferenceExpRunner import InferenceExpRunner


def sub_main(configs, model):
    print("Loading configuration, setting up output directories...")
    configs = load_config_and_setup_output_dir(configs)

    """Load the data"""
    raw_datasets = load_data_from_hf(configs.data_args.dataset_name, subset_name=configs.data_args.dataset_subset_name,
                                     split=configs.data_args.dataset_split_name, cache_dir=configs.data_args.cache_dir)

    """Preprocess data"""
    tokenized_datasets, tokenizer, verifier, short_dataset_name = preprocess(configs, raw_datasets)
    data_loaders = setup_dataloader(input_datasets=tokenized_datasets, batch_size=configs.running_args.batch_size,
                                    tokenizer=tokenizer)

    # """Load model"""
    # model = load_model(configs)

    """Set up trainer"""
    exp_runner = InferenceExpRunner(model=model, tokenizer=tokenizer, data_loader=data_loaders,
                                    inference_backend=configs.model_args.inference_backend,
                                    args=configs.running_args,
                                    enable_thinking=configs.model_args.enable_thinking,
                                    main_cot_keep_rate=configs.model_args.main_cot_keep_rate,
                                    data_source=short_dataset_name,
                                    think_start_str=configs.model_args.think_start_str,
                                    think_end_str=configs.model_args.think_end_str,
                                    monitor_model_name=configs.running_args.monitor_model_name,
                                    monitor_backend_type=configs.running_args.monitor_backend_type,
                                    judge_model_name=configs.running_args.llm_judge_model_name,
                                    judge_backend_type=configs.running_args.llm_judge_backend_type)

    try:
        if short_dataset_name == "general_reward":
            exp_runner.general_reward_inference_main_model(verifier=verifier)
        else:
            exp_runner.specific_behavior_inference_main_model(verifier=verifier)
        print("yay!")
    finally:
        # Clean up distributed process group if it was initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

@hydra.main(config_path="configs", config_name="inference_main_model", version_base=None)
def main(configs):
    """Load model"""
    model = load_model(configs)
    sub_main(configs, model)

if __name__ == "__main__":
    main()
