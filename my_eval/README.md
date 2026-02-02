# classmate_CoT

## Structure
```
└── configs
    ├── base_configs.yaml: contains the base configurations for the project
    ├── overwrite_configs.yaml: include variations that you want to overwrite to the base configurations
    └── ...
└── src
    ├── analysis_module: scripts for analysis
    ├── data_module: data loading and processing (tokenization...)
    ├── model_module: model definition and loading
    ├── train_module: code for training
    ├── (Optional) eval_module: code for evaluation
    └── ...
└── .gitignore
└── main.sh: main entry point for the project
└── requirements.txt: list of dependencies
└── ...
```

## Usage
### Setup
Follow instructions from classmate-cot-verl
[//]: # (```)
[//]: # (conda create -n cot_vllm python=3.10.12)
[//]: # (conda activate cot_vllm)
[//]: # (pip install -r requirements_vllm.txt)
[//]: # (```)

## Run Exps

### Utility Across RL Steps
1. Inference main model: [inference_main_model.sh](scripts/inference_main_model.sh)
2. Remove last steps from CoT: [utils_across_models.py](src/analysis_module/utils_across_models.py)
3. Inference CoT to classmate model: [cot_utility_to_classmate.sh](scripts/cot_utility_to_classmate.sh)
4. Evaluate classmate model: [visualize_cot_utility_across_RL_across_models.py](src/analysis_module/visualize_cot_utility_across_RL_across_models.py)
5. Main/classmate correctness analysis: [visualize_main_classmate_correctness.py](src/analysis_module/visualize_main_classmate_correctness.py)

