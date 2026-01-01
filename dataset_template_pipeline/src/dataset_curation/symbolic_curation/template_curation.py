import re
import os
import json
import torch
import requests
import pandas as pd  
from openpyxl import Workbook 
from pathlib import Path
from openai import OpenAI
from collections import Counter
from datasets import load_dataset
from typing import List, Dict, Optional, Union



#######################
# Helper Functions
#######################

# Function reads all json objects in a folder in order
def read_json_from_dir(directory_path: str):

    objects = []

    # Sort json files
    json_files = sorted(
        [f for f in os.listdir(directory_path) if f.endswith('.json')]
    )

    for file_name in json_files:
        # Skip 'ic_template.json'
        if file_name == "ic_template.json":
            continue

        file_path = os.path.join(directory_path, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                obj = json.loads(content)
                objects.append(obj)
        except Exception as e:
            print(f"failed reading {file_name}: {e}")
    
    return objects


# Given English template, it will extract the {key, value}
def variable_pairing_extraction(template: str):

    pattern = r"\{([^,{}]+),\s*([^{}]+)\}"
    matches = re.findall(pattern, template)

    variable_mapping = {}
    for var_name, var_value in matches:
        variable_mapping[var_name.strip()] = var_value.strip()
    return variable_mapping


# Load the first 'limit' number of dataset from hf
def dataset_loading(hf_data_path: str,
                    lang_code: str,
                    split: str,
                    limit: int):
    ds = load_dataset(hf_data_path, lang_code, split = split)

    question_list = []

    max_length = min(limit, len(ds))

    for example in ds.select(range(max_length)):
        question = example["question"]
        question_list.append(question)
    return question_list


# Retrieve the corresponding translation word in number_cache
def translate_number_word(number_cache: str, word: str) -> str:
    key = word.lower()
    return number_cache[key]

# Retrievce the list of translated words
def translate_list(number_cache: str, lst: list[str]) -> list[str]:
    return [translate_number_word(number_cache, w) for w in lst]


#######################
# Hyperparams
#######################
"""
ru te sw th
russian telugu swahili thai

amharic ewe hausa igbo kinyarwanda lingala luganda oromo shona sotho twi vai
amh twi  
"""
LANG_CODE = "twi"
LANGUAGE = "twi"

INPUT_FOLDER = "/home/mila/x/xut/github/afrimgsm-symbolic/template/english/IC"
#HF_DATASET = "juletxara/mgsm"
HF_DATASET = "masakhane/afrimgsm"

OUTPUT_PATH = "/home/mila/x/xut/github/afrimgsm-symbolic/template"
TEMPLATE_DATASET = "symbolic"

INSTRUCT_FILE_FILLER = "/home/mila/x/xut/github/afrimgsm-symbolic/prompt/template_construction/template_instruction_{lang}.txt"
INSTRUCT_FILE = INSTRUCT_FILE_FILLER.format(lang=LANG_CODE)
NUMBER_CACHE_DIR = "/home/mila/x/xut/github/afrimgsm-symbolic/multilingual_number_data"
NUMBER_CACHE_PATH = os.path.join(NUMBER_CACHE_DIR, LANGUAGE, "number_word_exact_map.json")

# Setup openai
client = OpenAI(api_key="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")
with open(INSTRUCT_FILE, "r", encoding="utf-8") as f:
    template_construction_instructions = f.read()



#######################
# Template Creation Logic
#######################

# Init empty lists
src_question_list = []
src_template_list = []
src_variable_list = []

tgt_question_list = []
tgt_template_list = []
tgt_variable_list = []


# Read json objects
src_json_objets = read_json_from_dir(INPUT_FOLDER)
print(type(src_json_objets))

# Loop through json objects
for obj in src_json_objets:

    print(type(obj))
    print(len(obj))

    src_question_list.append(obj["original_question"])

    template = obj["question_template"]
    src_template_list.append(template)

    variable_dict = variable_pairing_extraction(template)
    src_variable_list.append(variable_dict)

# Extract limit and load tgt language dataset
limit = len(src_question_list)

tgt_question_list = dataset_loading(hf_data_path=HF_DATASET,
                                    lang_code=LANG_CODE,
                                    split='test',
                                    limit=limit)


# Genearl loop
for idx, question in enumerate(tgt_question_list):

    # Construct request
    request = f"""
    Input: {src_template_list[idx]}
    Native: {question}
    Output: 
    """
    print(request)
    response = client.responses.create(
        model="gpt-4.1",
        instructions=template_construction_instructions,
        input=request,
    )

    tgt_question_template = response.output_text
    print("===================================")
    print(f"SUMMARY: idx {idx}")
    print(f"original question: {src_question_list[idx]}")
    print(f"original template: {src_template_list[idx]}")
    print(f"tgt question: {question}")
    print(f"tgt question_template {tgt_question_template}")

    tgt_template_list.append(tgt_question_template)

print(src_template_list)
print(tgt_template_list)


# Save the current list of tgt templates
template_filename = "template.json"
OUTPUT_FOLDER = os.path.join(OUTPUT_PATH, LANGUAGE, TEMPLATE_DATASET)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(os.path.join(OUTPUT_FOLDER, template_filename), "w", encoding="utf-8") as f:
    json.dump(tgt_template_list, f, ensure_ascii=False, indent=2)


# Export to excel
excel_path = Path(OUTPUT_PATH, LANGUAGE, "template.xlsx")
df = pd.DataFrame(
    {
        "english": src_template_list,
        LANGUAGE: tgt_template_list,
    }
)
df.to_excel(excel_path, index=False)            # uses openpyxl backend


#######################
# Construct the target json objects
#######################

# Load the number cache dir from tgt language

if os.path.exists(NUMBER_CACHE_PATH):
    with open(NUMBER_CACHE_PATH, "r", encoding="utf-8") as f:
        NUMBER_CACHE = json.load(f)
else:
    print("[ERROR] Translation number not available")


for idx, obj in enumerate(src_json_objets):

    # Create copy and replace template, question, numbervariable dict
    new_obj = obj.copy()

    new_obj["original_question"] = tgt_question_list[idx]
    new_obj["question_template"] = tgt_template_list[idx]

    # Update the variable number dictionary
    var_dict = new_obj.get("variable_number_dictionary", {})
    for k, v in var_dict.items():
        # Only care about lists, and it shoudl only be lists
        if not isinstance(v, list):
            print("[ERROR]: NON-LIST FOUND")
            continue
        
        if k.startswith("num_text_"):
            # Retrieve the translated list
            var_dict[k] = translate_list(number_cache=NUMBER_CACHE, lst=v)
        
        elif k.startswith("num_frac_"):
            frac_list = []

            # Keep the numerical fractions as is
            for item in v:
                if isinstance(item, str) and re.fullmatch(r"\d+/\d+", item.strip()):
                    frac_list.append(item)
                else:
                    frac_list.append(translate_list(number_cache=NUMBER_CACHE, lst=[item])[0])
            
            var_dict[k] = frac_list
    new_obj["variable_number_dictionary"] = var_dict


    # write file to folder
    filename = f"{idx:04d}.json"
    with open(os.path.join(OUTPUT_FOLDER, filename), "w", encoding="utf-8") as f:
        json.dump(new_obj, f, ensure_ascii=False, indent=2)










