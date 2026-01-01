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
        (f for f in os.listdir(directory_path)
         if f.endswith('.json') and f[:-5].isdigit()),
        key=lambda x: int(x[:-5])
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

LANG_CODE = "hau"
LANGUAGE = "hausa"
INPUT_FOLDER = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/english/IC"
TGT_INPUT_FOLDER = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/symbolic"

OUTPUT_PATH = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/IC"

INSTRUCT_INSERTION_FILE_FILLER = "/home/mila/x/xut/github/afrimgsm-symbolic/prompt/template_no_op_construction/template_insertion_{lang}.txt"
INSTRUCT_INSERTION_FILE = INSTRUCT_INSERTION_FILE_FILLER.format(lang=LANG_CODE)

# Setup openai
client = OpenAI(api_key="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")
with open(INSTRUCT_INSERTION_FILE, "r", encoding="utf-8") as f:
    template_insertion_instructions = f.read()


#######################
# IC Template Construction Logic
#######################

src_ic_num = []
tgt_original_question = []
tgt_ic_sentence = []
tgt_ic_template = []
tgt_symbolic_template = []

src_json_objects = read_json_from_dir(INPUT_FOLDER)
tgt_json_object = read_json_from_dir(TGT_INPUT_FOLDER)

for obj in src_json_objects:
    src_ic_num.append(obj["variable_number_dictionary"]['num_ic'])
for obj in tgt_json_object:
    tgt_original_question.append(obj["original_question"])
    tgt_symbolic_template.append(obj["question_template"])


OUTPUT_FOLDER = os.path.join(OUTPUT_PATH)
template_filename = "ic_template.json"
file_path = os.path.join(OUTPUT_FOLDER, template_filename)
with open(file_path, "r", encoding="utf-8") as f:
    tgt_ic_sentence_template = json.load(f)




for tgt_symbolic, tgt_ic_sentence in zip(tgt_symbolic_template, tgt_ic_sentence_template):

    print(tgt_symbolic)
    print(tgt_ic_sentence)


    # Construct request
    request = f"""
    Native template: {tgt_ic_sentence}
    Native sentence: {tgt_symbolic}
    Output: 
    """
    response = client.responses.create(
        model="gpt-5",
        instructions=template_insertion_instructions,
        input=request,
    )

    tgt_ic_template.append(response.output_text)
    print(response.output_text)

    print(request)
    print(response.output_text)



# Save the current list of tgt ic templates, this will be going through an annotation correction stage
template_filename = "ic_inserted_template.json"
OUTPUT_FOLDER = os.path.join(OUTPUT_PATH)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
with open(os.path.join(OUTPUT_FOLDER, template_filename), "w", encoding="utf-8") as f:
    json.dump(tgt_ic_template, f, ensure_ascii=False, indent=2)





#######################
# IC JSON Objection Construction Logic
#######################

idx = 0
for orig_obj, ic_template, ic_sentence in zip(tgt_json_object, tgt_ic_template, tgt_ic_sentence_template):

    orig_obj["ic_question_template"] = ic_template
    orig_obj["ic_sentence"] = ic_sentence
    orig_obj["original_question"] = tgt_original_question[idx]
    orig_obj["question_template"] = tgt_symbolic_template[idx]


    # write file to folder
    filename = f"{idx:04d}.json"
    with open(os.path.join(OUTPUT_FOLDER, filename), "w", encoding="utf-8") as f:
        json.dump(orig_obj, f, ensure_ascii=False, indent=2)

    idx = idx + 1













