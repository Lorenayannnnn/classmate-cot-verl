# from my_eval.src.data_module.dataset_objects.gpqa import GPQA

from my_eval.src.data_module.dataset_objects.aime import AIME
from my_eval.src.data_module.dataset_objects.gsm8k import GSM8K
from my_eval.src.data_module.dataset_objects.hendrycks_math import HendrycksMath
from verl.utils.reward_score.cot_monitor.MMLUSycophancyVerifier import MMLUSycophancyVerifier as MMLUSycophancy
from verl.utils.reward_score.cot_monitor.SycophancyVerifier import SycophancyVerifier

# from my_eval.src.data_module.dataset_objects.mmlu import MMLU

DATASET_NAME_TO_CLASS = {
    "gsm8k": GSM8K,
    "hendrycks_math": HendrycksMath,
    "aimo-validation-aime": AIME,
    # "gpqa": GPQA,
    # "mmlu": MMLU
    "mmlu_sycophancy": MMLUSycophancy,
    "helpful_instructions": SycophancyVerifier
    # "helpful_instructions": GeneralSycophancyVerifier
}

DATASET_NAME_TO_SUBSET_NAME_LIST = {
    "gsm8k": ["/main"],
    "hendrycks_math": ["/algebra", "/counting_and_probability", "/geometry", "/intermediate_algebra", "/number_theory", "/prealgebra", "/precalculus"],
    "aimo-validation-aime": ["/default"],
    "gpqa": ["/gpqa_main"],
    # excluded: professional_law
    "mmlu": ['/abstract_algebra', '/anatomy', '/astronomy', '/business_ethics', '/clinical_knowledge', '/college_biology', '/college_chemistry', '/college_computer_science', '/college_mathematics', '/college_medicine', '/college_physics', '/computer_security', '/conceptual_physics', '/econometrics', '/electrical_engineering', '/elementary_mathematics', '/formal_logic', '/global_facts', '/high_school_biology', '/high_school_chemistry', '/high_school_computer_science', '/high_school_european_history', '/high_school_geography', '/high_school_government_and_politics', '/high_school_macroeconomics', '/high_school_mathematics', '/high_school_microeconomics', '/high_school_physics', '/high_school_psychology', '/high_school_statistics', '/high_school_us_history', '/high_school_world_history', '/human_aging', '/human_sexuality', '/international_law', '/jurisprudence', '/logical_fallacies', '/machine_learning', '/management', '/marketing', '/medical_genetics', '/miscellaneous', '/moral_disputes', '/moral_scenarios', '/nutrition', '/philosophy', '/prehistory', '/professional_accounting', '/professional_medicine', '/professional_psychology', '/public_relations', '/security_studies', '/sociology', '/us_foreign_policy', '/virology', '/world_religions']
}