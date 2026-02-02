import argparse
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
    tokenized_datasets, tokenizer, dataset_object = preprocess(configs, raw_datasets)
    data_loaders = setup_dataloader(input_datasets=tokenized_datasets, batch_size=configs.running_args.batch_size,
                                    tokenizer=tokenizer)

    # """Load model"""
    # model = load_model(configs)

    """Set up trainer"""
    exp_runner = InferenceExpRunner(model=model, tokenizer=tokenizer, data_loader=data_loaders,
                                    inference_backend=configs.model_args.inference_backend, args=configs.running_args)

    try:
        # if configs.running_args.exp_type == "inference_main_model":
        exp_runner.inference_main_model(dataset_object=dataset_object)
        # elif configs.running_args.exp_type == "cot_utility_to_classmate":
        #     exp_runner.run_cot_utility_to_classmate(dataset_object=dataset_object, do_wo_cot=args.wo_cot)
        # else:
        #     raise ValueError(f"Unknown experiment type {configs.running_args.exp_type}")
        print("yay!")
    finally:
        # Clean up distributed process group if it was initialized
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

@hydra.main(config_path="configs", config_name="inference_main_model", version_base=None)
def main(configs):
    """Load model"""
    model = load_model(configs)
    if configs.data_args.dataset_name == "EleutherAI/hendrycks_math":
        subsets = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
        for subset in subsets:
            print(f"Processing subset: {subset}")
            configs.data_args.dataset_subset_name = subset
            sub_main(configs, model)
    else:
        sub_main(configs, model)

if __name__ == "__main__":
    main()
