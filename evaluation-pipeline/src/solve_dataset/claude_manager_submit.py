from anthropic import Anthropic
from pathlib import Path
import os
import yaml
from typing import List
import json
from argparse import ArgumentParser

from prompts.PROMPTS import *
from solve_dataset.claude_batch_math_manager import ClaudeBatchApiManager


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
    """Loads dataset for specified dataset-type + language"""

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
    """Compute the result directory path based on dataset type and language"""

    if dataset_type == "original-mgsm":
        dataset_path = os.path.join(result_basedir, "original_dataset", "MGSM", model_name, language) 
    elif dataset_type == "original-afrimgsm":
        dataset_path = os.path.join(result_basedir, "original_dataset", "AFRI_MGSM", model_name, language) 
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
    parser = ArgumentParser(description="Anthropic Evaluation runner")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load configuration
    base_dataset_folder = config["base_dataset_folder"]
    dataset_type = config["dataset_type"]
    language = config["language"]
    model_name = config["model_name"]  # e.g., "claude-sonnet-4-0"
    model_max_token = config["model_max_token"]
    model_temperature = config["model_temperature"]
    result_basedir = config["result_basedir"]
    anthropic_api_key = "anthropic_api_key"
    
    # Load all language datasets
    result_path_dir = {}
    for lang in language:
        if (config["dataset_type"] == "distribution-ic") or (config["dataset_type"] == "distribution-symbolic"):
            for i in range(config["total_num_variation"]):
                curr_data = load_dataset(base_dataset_folder, dataset_type, lang, index=i)
                result_path = compute_result_dir_path(result_basedir, dataset_type, lang, model_name, i)
                result_path_dir[result_path] = curr_data
        else:
            curr_data = load_dataset(base_dataset_folder, dataset_type, lang)
            result_path = compute_result_dir_path(result_basedir, dataset_type, lang, model_name)
            result_path_dir[result_path] = curr_data

    # Setup Anthropic client
    client = Anthropic(api_key=anthropic_api_key)
    
    # Process each dataset
    for result_dir_path, dataset_list in result_path_dir.items():
        
        # Initialize Claude Batch Manager
        manager = ClaudeBatchApiManager(
            client=client,
            base_dir=Path(result_dir_path),
            deployment=model_name
        )

        # Build prompts dictionary with custom IDs
        prompts_dict = {}
        
        # Use ONLY the third prompts, craft them
        # for prompt_num in [0, 1, 2, 3, 4, 5]:
        for prompt_num in [3]:
            question_only_list = [data["question"] for data in dataset_list]

            for q_idx, question in enumerate(question_only_list):
                var_name = f"cot_prompt_{prompt_num}"
                prompt_template = globals()[var_name]
                curr_prompt = prompt_template.format(question=question)
                
                # Create unique custom_id: question_index + prompt_num
                custom_id = f"q{q_idx}_prompt{prompt_num}"
                prompts_dict[custom_id] = curr_prompt
        
        out_dir = os.path.join(result_dir_path, "results")
        os.makedirs(out_dir, exist_ok=True)
        answers_jsonl = os.path.join(out_dir, "answers.jsonl")

        # Skip if results already exist
        if not os.path.exists(answers_jsonl):
            print(f"\n[PROCESSING] {result_dir_path}")
            
            # Run the batch computation
            batch_id, results = manager.compute_batch(
                custom_id_to_texts=prompts_dict,
                temperature=model_temperature,
                max_tokens=model_max_token,
                poll_interval=20,
                max_wait_seconds=7200,  # 2 hours timeout
                verbose=True,
                write_history=True,
                save_results=True
            )

            print(f"\n[BATCH COMPLETE] Batch ID: {batch_id}")
            
            # Write structured answers (jsonl), pairing question + answer
            with open(answers_jsonl, "w", encoding="utf-8") as f:
                for custom_id, result in results.items():
                    # Extract the answer text
                    answer_text = ""
                    if result["type"] == "succeeded":
                        content_blocks = result["message"]["content"]
                        text_blocks = [b.get("text") for b in content_blocks if b.get("type") == "text"]
                        if text_blocks:
                            answer_text = text_blocks[0]
                    else:
                        answer_text = f"ERROR: {result.get('error', 'Unknown error')}"
                    
                    # Get the original prompt
                    question = prompts_dict[custom_id]
                    
                    rec = {
                        "custom_id": custom_id,
                        "model": model_name,
                        "question": question,
                        "answer": answer_text,
                        "batch_id": batch_id,
                        "result_type": result["type"]
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
            print(f"[SAVED] Results written to {answers_jsonl}")
        else:
            print(f"[SKIPPING] - Path already exists {answers_jsonl}")


if __name__ == "__main__":
    main()