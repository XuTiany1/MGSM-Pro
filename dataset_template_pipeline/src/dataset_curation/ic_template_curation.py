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
        file_path = os.path.join(directory_path, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

                obj = json.loads(content)
                objects.append(obj)
        except Exception as e:
            print(f"failed reading {file_name}: {e}")
    
    return objects




#######################
# Hyperparams
#######################

LANG_CODE = "en"
LANGUAGE = "english"
INPUT_FOLDER = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/symbolic"
OUTPUT_PATH = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/IC"

INSTRUCT_FILE_FILLER = "/home/mila/x/xut/github/afrimgsm-symbolic/prompt/template_no_op_construction/template_instruction_{lang}.txt"
INSTRUCT_FILE = INSTRUCT_FILE_FILLER.format(lang=LANG_CODE)

# Setup openai
client = OpenAI(api_key="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")
with open(INSTRUCT_FILE, "r", encoding="utf-8") as f:
    template_construction_instructions = f.read()

#######################
# Template Creation Logic
#######################

src_template_list = []
src_solution_list = []
src_num_var_list = []
src_original_answer_list = []

ic_response_list = []
ic_template_list = []   # This is for excel purposes


src_json_objects = read_json_from_dir(INPUT_FOLDER)

# loop through and append
for obj in src_json_objects:

    src_template_list.append(obj['question_template'])
    src_solution_list.append(obj['question_solution'])
    src_original_answer_list.append(obj['original_answer'])
    src_num_var_list.append(obj['variable_number_dictionary'])



for i in range(len(src_template_list)):

    curr_template = src_template_list[i]
    curr_solution = src_solution_list[i]
    curr_original_ans = src_original_answer_list[i]
    curr_num_var_dict = src_num_var_list[i]

    # Construct request
    request = f"""
    question_template: {curr_template}
    question_solution: {curr_solution}
    variable_number_dictionary: {curr_num_var_dict}
    original_answer: {curr_original_ans}
    """

    response = client.responses.create(
        model="gpt-4.1",
        instructions=template_construction_instructions,
        input=request,
    )
    # Append so that it is dict
    ic_response_list.append(json.loads(response.output_text))
    ic_template_list.append(json.loads(response.output_text)["ic_sentence"])

# Save the current list of tgt templates
template_filename = "ic_template.json"
OUTPUT_FOLDER = os.path.join(OUTPUT_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(os.path.join(OUTPUT_FOLDER, template_filename), "w", encoding="utf-8") as f:
    json.dump(ic_response_list, f, ensure_ascii=False, indent=2)


# Export to excel
excel_path = Path(OUTPUT_PATH,"template.xlsx")
df = pd.DataFrame(
    {
        "symbolic template": src_template_list,
        "IC template": ic_template_list,
    }
)
df.to_excel(excel_path, index=False)

idx = 0
for orig_obj, ic_obj in zip(src_json_objects, ic_response_list):
    orig_obj["ic_question_template"] = ic_obj["ic_question_template"]
    orig_obj["ic_sentence"] = ic_obj["ic_sentence"]

    # Add num_ic (or ic_numbers) INSIDE variable_number_dictionary
    if "variable_number_dictionary" in orig_obj:
        if "num_ic" in ic_obj:
            orig_obj["variable_number_dictionary"]["num_ic"] = ic_obj["num_ic"]
        else:
            orig_obj["variable_number_dictionary"]["num_ic"] = ic_obj["ic_numbers"]
    else:
        print("Warning: variable_number_dictionary missing in one entry!")


    # write file to folder
    filename = f"{idx:04d}.json"
    with open(os.path.join(OUTPUT_FOLDER, filename), "w", encoding="utf-8") as f:
        json.dump(orig_obj, f, ensure_ascii=False, indent=2)

    idx = idx + 1



# Save merged list to a new file
#merged_filename = "merged_with_ic.json"
#with open(os.path.join(OUTPUT_FOLDER, merged_filename), "w", encoding="utf-8") as f:
#    json.dump(src_json_objects, f, ensure_ascii=False, indent=2)



































