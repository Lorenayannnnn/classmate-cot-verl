"""
Minimal RL training loop using GRPO-style reward centering.

Variable naming convention (see CONTRIBUTING.md):
    _P: Problem dimension (different questions/prompts in a batch)
    _G: Group dimension (multiple rollouts per problem for variance reduction)
    _T: Token/Time dimension (sequence positions)
    _D: Datum dimension (training examples after flattening)

Example: `tokens_G_T` is a list of token sequences, one per group member.
In this script, datums_D has size P*G (one datum per rollout).
"""

import logging
import os

import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# MODEL_NAME = "Qwen/Qwen3-4B"
# MODEL_NAME = "Qwen/Qwen3-1.7B"
# MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME = "Qwen/Qwen3-Coder-Next"
# MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B"
DATA_DIR = "data/code_contests_pass_wrong_sol"
DATA_SPLIT = "train"  # "dev" or "test" for fixed splits
SEED = 42
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 3072

# """Generate one test case for the following coding problem.\n{test}\n\nReturn only your reasoning and the test case in the following format:\n### Reasoning \n{{your reasoning}}\n\n### Test Case\n```json\n{{"input": "...", "output": "..."}}\n```"""


def load_dataset_split(data_dir: str, split: str, seed: int = 42) -> datasets.Dataset:
    if split == "train":
        parquet_path = os.path.join(data_dir, f"seed_{seed}", "train.parquet")
    else:
        parquet_path = os.path.join(data_dir, f"{split}.parquet")
    return datasets.load_dataset("parquet", data_files=parquet_path)["train"]


def main():
    logging.basicConfig(level=logging.INFO)

    # Initialize vLLM engine
    logger.info(f"Initializing vLLM engine for {MODEL_NAME}...")
    llm = LLM(model=MODEL_NAME,max_model_len=MAX_MODEL_LEN,gpu_memory_utilization=0.9,enable_prefix_caching=True,trust_remote_code=True,)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    sampling_params = SamplingParams(temperature=1,top_p=1,max_tokens=MAX_NEW_TOKENS,)

    # Load dataset
    logger.info(f"Loading dataset from {DATA_DIR} (split={DATA_SPLIT})...")
    dataset = load_dataset_split(DATA_DIR, DATA_SPLIT, seed=SEED)
    logger.info(f"Loaded {len(dataset)} examples.")

    for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Inference"):
        prompt_messages = entry["prompt"]  # list of {"role": ..., "content": ...}
        ground_truth = entry["reward_model"]["ground_truth"]
        description = entry["reward_model"]["raw_question_for_monitor"]

        question = """Dr. Asimov, a robotics researcher, released cleaning robots he developed (see Problem B). His robots soon became very popular and he got much income. Now he is pretty rich. Wonderful.\n\nFirst, he renovated his house. Once his house had 9 rooms that were arranged in a square, but now his house has N × N rooms arranged in a square likewise. Then he laid either black or white carpet on each room.\n\nSince still enough money remained, he decided to spend them for development of a new robot. And finally he completed.\n\nThe new robot operates as follows:\n\n* The robot is set on any of N × N rooms, with directing any of north, east, west and south.\n* The robot detects color of carpets of lefthand, righthand, and forehand adjacent rooms if exists. If there is exactly one room that its carpet has the same color as carpet of room where it is, the robot changes direction to and moves to and then cleans the room. Otherwise, it halts. Note that halted robot doesn\'t clean any longer. Following is some examples of robot\'s movement.\n\n<image>\n\nFigure 1. An example of the room\n\n\nIn Figure 1,\n* robot that is on room (1,1) and directing north directs east and goes to (1,2).\n* robot that is on room (0,2) and directing north directs west and goes to (0,1).\n* robot that is on room (0,0) and directing west halts.\n* Since the robot powered by contactless battery chargers that are installed in every rooms, unlike the previous robot, it never stops because of running down of its battery. It keeps working until it halts.\n\n\n\nDoctor\'s house has become larger by the renovation. Therefore, it is not efficient to let only one robot clean. Fortunately, he still has enough budget. So he decided to make a number of same robots and let them clean simultaneously.\n\nThe robots interacts as follows:\n\n* No two robots can be set on same room.\n* It is still possible for a robot to detect a color of carpet of a room even if the room is occupied by another robot.\n* All robots go ahead simultaneously.\n* When robots collide (namely, two or more robots are in a single room, or two robots exchange their position after movement), they all halt. Working robots can take such halted robot away.\n\n\n\nOn every room dust stacks slowly but constantly. To keep his house pure, he wants his robots to work so that dust that stacked on any room at any time will eventually be cleaned.\n\nAfter thinking deeply, he realized that there exists a carpet layout such that no matter how initial placements of robots are, this condition never can be satisfied. Your task is to output carpet layout that there exists at least one initial placements of robots that meets above condition. Since there may be two or more such layouts, please output the K-th one lexicographically.\n\nConstraints\n\n* Judge data consists of at most 100 data sets.\n* 1 ≤ N < 64\n* 1 ≤ K < 263\n\nInput\n\nInput file contains several data sets. One data set is given in following format:\n\n\nN K\n\n\nHere, N and K are integers that are explained in the problem description.\n\nThe end of input is described by a case where N = K = 0. You should output nothing for this case.\n\nOutput\n\nPrint the K-th carpet layout if exists, "No" (without quotes) otherwise.\n\nThe carpet layout is denoted by N lines of string that each has exactly N letters. A room with black carpet and a room with white carpet is denoted by a letter \'E\' and \'.\' respectively. Lexicographically order of carpet layout is defined as that of a string that is obtained by concatenating the first row, the second row, ..., and the N-th row in this order.\n\nOutput a blank line after each data set.\n\nExample\n\nInput\n\n2 1\n2 3\n6 4\n0 0\n\n\nOutput\n\n..\n..\n\nNo\n\n..EEEE\n..E..E\nEEE..E\nE..EEE\nE..E..\nEEEE..\n\n"""

        prompt_template = """Generate one test case for the following coding problem.\n{test}\n\nReturn only your reasoning and the test case in the following format:\n### Reasoning \n{{your reasoning}}\n\n### Test Case\n```json\n{{"input": "...", "output": "..."}}\n```"""
        # messages = [{"role": "system", "content": f"You are an expert programmer that generate your chain-of-thought reasoning wrapped with <think></think> and test case in the specified format."},{"role": "user", "content": f"{prompt_template.format(test=question)}"},]
        messages = [{"role": "user", "content": f"{prompt_template.format(test=question)}"},]
        # messages = [{"role": "user", "content": question}]

        # Apply chat template

        # Run inference
        # print(f"[{idx}] INPUT:\n{formatted_prompt}")
        print(llm.generate(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True), sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text)
        # print(f"\n{'='*60}")
        # print(f"OUTPUT:\n{generated_text}")
        breakpoint()


if __name__ == "__main__":
    main()
