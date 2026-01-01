from openai import AzureOpenAI

import os
import yaml
from typing import List
import json
from argparse import ArgumentParser

from prompts.PROMPTS import *
from solve_dataset.batch_math_manager import BatchMathManager



def construct_prompt(questions: List[str], prompt_num: int):
    """Construct all six prompts"""
    final_list_of_prompt = []

    var_name = f"cot_prompt_{prompt_num}"

    for question in questions:
        if var_name in globals():
            prompt_template = globals()[var_name]
            curr_prompt = prompt_template.format(question=question)
            final_list_of_prompt.append(curr_prompt)

    return final_list_of_prompt


def load_dataset(basefolder: str, dataset_type: str, language: str, index=0):
    """ Loads dataset for specified dataset-type + langauge"""

    if dataset_type == "original-mgsm":
        dataset_path = os.path.join(basefolder, "original_dataset", "MGSM", language, "test.jsonl") 
    elif dataset_type == "original-afrimgsm":
        dataset_path = os.path.join(basefolder, "original_dataset", "AFRI_MGSM", language, "test.jsonl") 
    elif dataset_type == "ic-names":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "IC", f"{language}_ic_names_rank1.jsonl")
    elif dataset_type == "ic-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "IC", f"{language}_ic_numbers_rank1.jsonl")
    elif dataset_type == "ic-names-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "IC", f"{language}_ic_names_numbers_rank1.jsonl") 
    elif dataset_type == "symbolic-names":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "symbolic", f"{language}_symbolic_names_rank1.jsonl")
    elif dataset_type == "symbolic-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "symbolic", f"{language}_symbolic_numbers_rank1.jsonl")
    elif dataset_type == "symbolic-names-numbers":
        dataset_path = os.path.join(basefolder, "hardest_variation_dataset", language, "symbolic", f"{language}_symbolic_names_numbers_rank1.jsonl") 
    elif dataset_type == "distribution-ic":
        dataset_path = os.path.join(basefolder, "distribution_dataset", language, "IC", f"distribution_dataset_{index}.jsonl") 
    elif dataset_type == "distribution-symbolic":
        dataset_path = os.path.join(basefolder, "distribution_dataset", language, "symbolic", f"distribution_dataset_{index}.jsonl") 
    elif dataset_type == "debug":
        dataset_path = "/home/mila/x/xut/github/evaluation-pipeline/dataset/debug/test.jsonl"

    print(dataset_path)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data


def compute_result_dir_path(result_basedir: str, dataset_type: str, language: str, model_name: str, index=0):

    if dataset_type == "original-mgsm":
        dataset_path = os.path.join(result_basedir, "original_dataset", "MGSM", model_name, language,) 
    elif dataset_type == "original-afrimgsm":
        dataset_path = os.path.join(result_basedir, "original_dataset","AFRI_MGSM", model_name, language) 
    elif dataset_type == "ic-names":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "names")
    elif dataset_type == "ic-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "numbers")
    elif dataset_type == "ic-names-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "IC", "names-numbers") 
    elif dataset_type == "symbolic-names":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "names")
    elif dataset_type == "symbolic-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "numbers")
    elif dataset_type == "symbolic-names-numbers":
        dataset_path = os.path.join(result_basedir, "hardest_variation_dataset", model_name, language, "symbolic", "names-numbers") 
    elif dataset_type == "distribution-ic":
        dataset_path = os.path.join(result_basedir, "distribution_dataset", model_name, language, "IC", f"distribution_result_{index}") 
    elif dataset_type == "distribution-symbolic":
        dataset_path = os.path.join(result_basedir, "distribution_dataset", model_name, language, "symbolic", f"distribution_result_{index}") 

    elif dataset_type == "debug":
        dataset_path = "/home/mila/x/xut/github/evaluation-pipeline/result/debug"

    return dataset_path


def main():

    # Argument parser
    # Load YAML files
    parser = ArgumentParser(description="Evaluation runner")
    parser.add_argument( "--config",  type=str,  required=True,  help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Launch multiple jobs
    base_dataset_folder = config["base_dataset_folder"]
    dataset_type = config["dataset_type"]
    language = config["language"]
    model_platform = config["model_platform"]
    model_name = config["model_name"]
    model_max_token = config["model_max_token"]
    model_temperature = config["model_temperature"]
    result_basedir = config["result_basedir"]
    thinking_budget = int(config.get("thinking_budget") or 0)

    
    # Load all language dataset into this array
    result_path_dir = {}
    for lang in language:

        if (config["dataset_type"] == "distribution-ic") or (config["dataset_type"] == "distribution-symbolic"):
            for i in range(config["total_num_variation"]):
                curr_data = load_dataset(base_dataset_folder, dataset_type, lang, index=i)

                result_path = compute_result_dir_path(result_basedir, dataset_type, lang, model_name, i)
                result_path_dir[result_path] = curr_data # List of Lists, each List is a distribution of digits

        else:
            curr_data = load_dataset(base_dataset_folder, dataset_type, lang) # A List
            result_path = compute_result_dir_path(result_basedir, dataset_type, lang, model_name)
            result_path_dir[result_path] = curr_data  # List

    # Setup client
    client = AzureOpenAI(
        api_version="api_version",
        azure_endpoint="azure_endpoint",
        api_key="api_key",
    )
    
    for result_dir_path, dataset_list in result_path_dir.items():

        mgr = BatchMathManager(
            client=client,
            base_dir=result_dir_path,
            deployment=model_name,
        )


        curr_question_batch = [] # Will be 250 questions

        # Use the 6 prompts, craft them
        # for prompt_num in [0, 1, 2, 3, 4, 5]:


            
        # Only use the 3rd prompts, craft them
        for prompt_num in [3]:
        
            question_only_list = [data["question"] for data in dataset_list]

            for question in question_only_list:
                var_name = f"cot_prompt_{prompt_num}"
                prompt_template = globals()[var_name]
                curr_prompt = prompt_template.format(question=question)
                curr_question_batch.append(curr_prompt)
            
        out_dir = os.path.join(result_dir_path, "results")
        os.makedirs(out_dir, exist_ok=True)
        answers_jsonl = os.path.join(out_dir, "answers.jsonl")

        # Skip those where this path already exist
        if not os.path.exists(answers_jsonl):
            result, stats = mgr.compute_math_batch(
                texts=curr_question_batch,
                temperature=model_temperature,
                max_tokens=model_max_token,
                check_interval=20,
                quiet=False,
            )


            
            # 4) write structured answers (jsonl), pairing question + answer
            answers_jsonl = os.path.join(out_dir, "answers.jsonl")
            with open(answers_jsonl, "w", encoding="utf-8") as f:
                for i, (q, a) in enumerate(zip(curr_question_batch, result)):
                    rec = {
                        "idx": i,
                        "model": model_name,          # your config model_name label
                        "deployment": model_platform, # if you want to record which Azure deployment you used
                        "question": q,
                        "answer": a,
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        else:
            print(f"[SKIPPING] - Path already exist {answers_jsonl}")


if __name__ == "__main__":
    main()





