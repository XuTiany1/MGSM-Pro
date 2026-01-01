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
    rng: random.Random,
    used_names=None
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """
    For each numeric combination, render one question.
    Returns:
      - generated_question: {question_text: answer}
      - name_maps: {question_text: {placeholder_key: chosen_name}}
    """
    generated_question: Dict[str, Any] = {}

    # Symbolic dataset should use the same name as IC, hence allow passing previously selected names
    if used_names != None:
        used_names = used_names
    else:
        used_names: Dict[str, str] = {}

    for combination in question_number_combinations:

        taken_values: Set[str] = set()

        def pick_name(base_key: str, default_val: str) -> str:
            pool = names.get(base_key, [])
            if not change_names or not pool:
                return default_val
            rng.shuffle(pool)
            for cand in pool:
                if cand not in taken_values:
                    taken_values.add(cand)
                    return cand
            # fallback: cycle
            val = pool[(len(taken_values)) % len(pool)]
            taken_values.add(val)
            return val

        def replace(match: re.Match) -> str:
            key = match.group(1)          # e.g. "num_digit_1" or "name_female_1"
            default_val = match.group(2)  # default text inside the braces

            # digits / general variables from THIS combination
            if key in combination:
                return str(combination[key])

            # name variables
            if key.startswith("name_"):
                # reuse same chosen name if placeholder appears again
                if key in used_names:
                    return used_names[key]

                # normalize base pool key: name_female_1 -> name_female
                base = key
                m = re.match(r"^(name_[a-zA-Z]+)(?:_\d+)?$", key)
                if m:
                    base = m.group(1)

                chosen = pick_name(base, default_val)
                used_names[key] = chosen
                return chosen

            # otherwise, stick with default
            return default_val

        question_text = PLACEHOLDER_RE.sub(replace, question_template)

        # ensure answer exists
        if "answer" not in combination:
            raise ValueError("Each combination must include an 'answer' field.")

        generated_question[question_text] = combination["answer"]
        used_names

    return generated_question, used_names



def process_yaml(yaml_filepath: str) -> Dict[str, Any]:

    with open(yaml_filepath, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = [
        "number_combination_filepath",
        "number_of_dataset_variation",
        "change_names",
        "languages",
        "language_name_folderpath",
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing required YAML keys: {missing}")

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

    for obj in template_json_obj:
        language_ic_templates.append(obj["ic_question_template"])
        language_symbolic_templates.append(obj["question_template"])

    return language_ic_templates, language_symbolic_templates


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

    rng = random.Random(42)


    # Load the question_number_combinations from path and ensures each contains 10 variations for each question
    with open(cfg["number_combination_filepath"], "r", encoding="utf-8") as f:
        number_combination = json.load(f)
    """
    number_combination here is a list, each element is a dictionary that contains the 'id': xxx, 'combination_1', ... 'combination_xx'.
    """
    # Verify that each question has 10 variations
    for i in range(len(number_combination)):
        # Subtract 1 to account for the 'id'
        combination_len = len(number_combination[i]) - 1 

        if combination_len != cfg["number_of_dataset_variation"]:
            print(f"ERROR: question id {i} only have {combination_len} out of the required {len(number_combination[i]) - 1} number variations")



    # Loop through each language
    for lang in (cfg["languages"]):

        # Load the language specific name data
        language_namedata = load_language_data(language=lang, language_name_folderpath=cfg["language_name_folderpath"])

        # Load the language specific dataset (both IC and Symbolic)
        language_ic_templates, language_symbolic_templates = load_language_template(language=lang, language_template_folderpath=cfg["template_root_folder"])

        print(language_ic_templates)

        # Init the empty language dataset
        distribution_ic_dataset = []
        distribution_symbolic_dataset = []

        # Generate the new dataset now
        for i in range(len(number_combination)):
            
            # Load the templates
            ic_template = language_ic_templates[i]
            symbolic_template = language_symbolic_templates[i]

            # Load number combination and remove the first 'id' entry
            number_comb = number_combination[i]

            
            question_num_combination = []
            for key, value in number_comb.items():
                if key != 'id':
                    question_num_combination.append(value)

            # Craft dataset
            ic_dataset, name_maps = craft_single_question_dataset(
                question_num_combination, ic_template, language_namedata, cfg["change_names"], rng
            )
            symbolic_dataset, name_maps = craft_single_question_dataset(
                question_num_combination, symbolic_template, language_namedata, cfg["change_names"], rng, used_names=name_maps
            )


            instance = 0
            for key, value in ic_dataset.items():
                ic_item = {
                    "id": i,
                    "instance": instance,
                    "question": key,
                    "answer": value
                }
                distribution_ic_dataset.append(ic_item)
                instance = instance + 1
            instance = 0
            for key, value in symbolic_dataset.items():
                ic_item = {
                    "id": i,
                    "instance": instance,
                    "question": key,
                    "answer": value
                }
                distribution_symbolic_dataset.append(ic_item)
                instance = instance + 1

            print("hi")

        write_to_dataset(lang , cfg["output_root_folder"], distribution_ic_dataset, distribution_symbolic_dataset, cfg["number_of_dataset_variation"])





    """
    question_number_combinations = [
        {"num_digit_1": 12, "num_digit_2": 65, "answer": 144},
        {"num_digit_1": 0, "num_digit_2": 0, "answer": 0},
    ]
    question_template = (
        "{name_female_1, Janet}’s ducks lay {num_digit_1, 16} eggs {name_female_2, Janet} per day. "
        "She eats {num_text_1, three} for breakfast every morning and bakes muffins for her friends "
        "every day with {num_text_2, four}. She sells the remainder at the farmers' market daily for "
        "${num_digit_2, 2} per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    )
    names = {
        "name_male": ["Bob", "James", "William", "Henry"],
        "name_female": ["Emily", "Charlotte", "Sophie", "Grace"],
    }
    change_names = True
    rng = random.Random(42)

    rendered, used_names = craft_single_question_dataset(
        question_number_combinations, question_template, names, change_names, rng
    )

    # pretty print
    for q, ans in rendered.items():
        print("Q:", q)
        print("A:", ans)
        print("Names used:", used_names)
        print("-" * 80)
    """

