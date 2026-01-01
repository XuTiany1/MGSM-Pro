from __future__ import annotations
import random
import re
import os
import yaml
import json
import argparse as arg
from itertools import islice
from typing import Dict, List, Tuple, Any, Set

PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z0-9_]+)\s*,\s*([^{}]+?)\}")

def craft_single_question_dataset(
    question_number_combinations: List[Dict[str, Any]],
    question_template: str,
    names: Dict[str, List[str]],
    change_names: bool,
    change_numbers: bool,
    rng: random.Random,
    original_answer: Any = None,
    used_names=None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]: # Changed Dict to List
    
    # Change this to a list
    generated_questions_list: List[Dict[str, Any]] = [] 
    name_maps_list: List[Dict[str, str]] = []

    if used_names is not None:
        used_names_list = used_names
    else:
        used_names_list = []

    for idx, combination in enumerate(question_number_combinations):
        if idx < len(used_names_list):
            current_names = used_names_list[idx]
            reusing_names = True
        else:
            current_names = {}
            reusing_names = False

        def pick_name(base_key: str, default_val: str) -> str:
            pool = names.get(base_key, [])
            if not change_names or not pool:
                return default_val
            return rng.choice(pool)

        def replace(match: re.Match) -> str:
            key = match.group(1)
            default_val = match.group(2)

            if key.startswith("num_"):
                if change_numbers and key in combination:
                    return str(combination[key])
                return default_val

            if key.startswith("name_"):
                if key in current_names:
                    return current_names[key]

                base = key
                m = re.match(r"^(name_[a-zA-Z]+)(?:_\d+)?$", key)
                if m:
                    base = m.group(1)

                chosen = pick_name(base, default_val)
                current_names[key] = chosen # Store it so it's consistent WITHIN this variation
                return chosen

            if key in combination:
                return str(combination[key])

            return default_val

        question_text = PLACEHOLDER_RE.sub(replace, question_template)

        if change_numbers:
            answer = combination["answer"]
        else:
            answer = original_answer

        # Append as an object instead of using a dictionary key
        generated_questions_list.append({
            "question": question_text,
            "answer": answer
        })
        
        if not reusing_names:
            name_maps_list.append(current_names)

    return generated_questions_list, name_maps_list


def process_yaml(yaml_filepath: str) -> Dict[str, Any]:
    with open(yaml_filepath, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = [
        "number_combination_filepath",
        "number_of_dataset_variation",
        "categories_to_change",
        "languages",
        "language_name_folderpath",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required YAML keys: {missing}")

    # Set default for skip list if not present
    if "problem_index_to_skip" not in cfg:
        cfg["problem_index_to_skip"] = []

    return cfg


def load_language_data(language: str, language_name_folderpath: str) -> Dict:

    # Assume all language data will be named language_names.json
    language_filename = language + "_words.json"
    language_filepath = os.path.join(language_name_folderpath, language_filename)

    with open(language_filepath, "r", encoding="utf-8") as f:
        language_namedata = json.load(f)

    return language_namedata

# Function reads all json objects in a folder in order
def read_json_from_dir(directory_path: str):

    objects = []

    # Sort json files
    json_files = sorted(
        [f for f in os.listdir(directory_path) if f.endswith('.json')]
    )

    for file_name in json_files:
        file_path = os.path.join(directory_path, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                obj = json.loads(content)
                objects.append(obj)
        except Exception as e:
            print(f"failed reading {file_name}: {e}")
    
    return objects


def load_language_template(language: str, language_template_folderpath: str):

    # Assume all language template will be named after 'name'
    folderpath = os.path.join(language_template_folderpath, language)
    template_json_obj = read_json_from_dir(folderpath)

    language_ic_templates = []
    language_symbolic_templates = []
    language_original_answers = []

    for obj in template_json_obj:
        language_ic_templates.append(obj["ic_question_template"])
        language_symbolic_templates.append(obj["question_template"])
        language_original_answers.append(obj.get("original_answer", None))

    return language_ic_templates, language_symbolic_templates, language_original_answers


def write_to_dataset(language: str, output_root_folder: str, distribution_ic_dataset: List, distribution_symbolic_dataset: List, num_variation: int):

    # Assume the output should be in the folder that is named 'language'
    output_folder_path = os.path.join(output_root_folder, language)
    os.makedirs(output_folder_path, exist_ok=True)

    # Craft dataset folders
    output_ic_folder_path = os.path.join(output_folder_path, "IC")
    output_symbolic_folder_path = os.path.join(output_folder_path, "symbolic")
    os.makedirs(output_ic_folder_path, exist_ok=True)
    os.makedirs(output_symbolic_folder_path, exist_ok=True)


    # Craft dataset by looping through all instance
    for i in range(num_variation):

        # Extract the IC and symbolic questions
        ic_dataset = []
        symbolic_dataset = []

        
        for ic_question, symb_question in zip(distribution_ic_dataset, distribution_symbolic_dataset):

            if ic_question["instance"] == i:
                ic_dataset.append(ic_question)
            if symb_question["instance"] == i:
                symbolic_dataset.append(symb_question)
        
        

        # Define output file paths for this variation
        ic_output_path = os.path.join(output_ic_folder_path, f"distribution_dataset_{i}.jsonl")
        symb_output_path = os.path.join(output_symbolic_folder_path, f"distribution_dataset_{i}.jsonl")

        # Write IC dataset
        with open(ic_output_path, "w", encoding="utf-8") as f:
            for entry in ic_dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Write symbolic dataset
        with open(symb_output_path, "w", encoding="utf-8") as f:
            for entry in symbolic_dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")





if __name__ == "__main__":

    # Actual main function
    parser = arg.ArgumentParser(description='generate dataset')
    parser.add_argument('--config', type=str, help='path to yaml file')
    args = parser.parse_args()  

    cfg = process_yaml(args.config)

    # Convert skip list to a set for efficient lookup
    skip_indices = set(cfg.get("problem_index_to_skip", []))

    rng = random.Random(42)

    # Determine what to change
    change_names = "names" in cfg["categories_to_change"]
    change_numbers = "numbers" in cfg["categories_to_change"]

    print(f"Configuration: change_names={change_names}, change_numbers={change_numbers}")

    # Load the question_number_combinations from path
    with open(cfg["number_combination_filepath"], "r", encoding="utf-8") as f:
        number_combination = json.load(f)
    
    """
    number_combination here is a list, each element is a dictionary that contains the 'id': xxx, 'combination_1', ... 'combination_xx'.
    """
    
    # If only changing names, we don't need to verify number combinations
    if change_numbers:
        # Verify that each question has the right number of variations
        for i in range(len(number_combination)):
            # Subtract 1 to account for the 'id'
            combination_len = len(number_combination[i]) - 1 

            if combination_len != cfg["number_of_dataset_variation"]:
                print(f"WARNING: question id {i} has {combination_len} variations, expected {cfg['number_of_dataset_variation']}")

    # Loop through each language
    for lang in (cfg["languages"]):

        # Load the language specific name data
        language_namedata = load_language_data(language=lang, language_name_folderpath=cfg["language_name_folderpath"])

        # Load the language specific dataset (both IC and Symbolic)
        language_ic_templates, language_symbolic_templates, language_original_answers = load_language_template(language=lang, language_template_folderpath=cfg["template_root_folder"])

        print(f"Processing language: {lang}")
        print(f"Number of templates: {len(language_ic_templates)}")

        # Init the empty language dataset
        distribution_ic_dataset = []
        distribution_symbolic_dataset = []

        # Generate the new dataset now
        for i in range(len(number_combination)):
            
            # Check if I need skipping
            if i in skip_indices:
                print(f"Skipping problem index {i} as requested in config.")
                continue
            
            # Load the templates
            ic_template = language_ic_templates[i]
            symbolic_template = language_symbolic_templates[i]
            original_answer = language_original_answers[i]

            # Load number combination and remove the first 'id' entry
            number_comb = number_combination[i]

            
            question_num_combination = []
            
            if change_numbers:
                # Use the number combinations from file
                for key, value in number_comb.items():
                    if key != 'id':
                        question_num_combination.append(value)
            else:
                # Not changing numbers - create empty combinations
                # The numbers will come from template defaults
                # We still need N variations for name changes
                for var_idx in range(cfg["number_of_dataset_variation"]):
                    # Empty dict means: use all defaults from template
                    # Just need a placeholder answer field
                    question_num_combination.append({"variation_id": var_idx})

            # Craft dataset
            ic_dataset, name_maps = craft_single_question_dataset(
                question_num_combination, ic_template, language_namedata, change_names, change_numbers, rng, original_answer
            )
            symbolic_dataset, _ = craft_single_question_dataset(
                question_num_combination, symbolic_template, language_namedata, change_names, change_numbers, rng, original_answer, used_names=name_maps
            )

            # Updated iteration logic
            for instance_idx, item in enumerate(ic_dataset):
                ic_item = {
                    "id": i,
                    "instance": instance_idx,
                    "question": item["question"],
                    "answer": item["answer"]
                }
                distribution_ic_dataset.append(ic_item)
            
            for instance_idx, item in enumerate(symbolic_dataset):
                symb_item = {
                    "id": i,
                    "instance": instance_idx,
                    "question": item["question"],
                    "answer": item["answer"]
                }
                distribution_symbolic_dataset.append(symb_item)
        print(f"Generated {len(distribution_ic_dataset)} IC questions and {len(distribution_symbolic_dataset)} symbolic questions for {lang}")

        write_to_dataset(lang, cfg["output_root_folder"], distribution_ic_dataset, distribution_symbolic_dataset, cfg["number_of_dataset_variation"])