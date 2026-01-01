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



#################
# Helper Functions
#################

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





#################
# Hypeparam
#################
INPUT_FOLDER = "/home/mila/x/xut/github/afrimgsm-symbolic/template/english/symbolic"
HF_DATASET = "juletxara/mgsm"
HF_DATASET = "masakhane/afrimgsm"

OUTPUT_PATH       = "/home/mila/x/xut/github/afrimgsm-symbolic/multilingual_number_data"
ENG_CACHE_PATH        = os.path.join(OUTPUT_PATH, "english", "number_word_exact_map.json")

# Setup openai
client = OpenAI(api_key="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")





#################
# Main Logic
#################

list_of_languages = ["yoruba", "hausa", "ewe", "amharic", "swahili", "igbo", "twi", "german", "japanese"]

numbers_set = set()


# Stage 1: I extract all english number texts that needs to be translated
src_json_objects = read_json_from_dir(INPUT_FOLDER)

for obj in src_json_objects:
    curr_number_dict = obj["variable_number_dictionary"]
    for key, value_list in curr_number_dict.items():
        for curr_val in value_list:

            if isinstance(curr_val, int) or isinstance(curr_val, float):
                continue

            elif isinstance(curr_val, str):
                # Check if it's a simple fraction
                curr_val = curr_val.lower()
                parts = curr_val.split('/')
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    continue  # skip fractions like '2/3'

                numbers_set.add(curr_val)

# Dump everything in english first
numbers_list = sorted(list(numbers_set))
os.makedirs(os.path.dirname(ENG_CACHE_PATH), exist_ok=True)
with open(ENG_CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(numbers_list, f, ensure_ascii=False, indent=2)



# Stage 2: I translate
for lang in list_of_languages:
    tgt_cache = Path(OUTPUT_PATH, lang, "number_word_exact_map.json")
    excel_path = tgt_cache.with_suffix(".xlsx")     # <— same base name

    if tgt_cache.exists():
        translations = json.loads(tgt_cache.read_text())
        print(f"[{lang}] JSON cache exists → {tgt_cache}")
    else:
        translations = {}
        for word in numbers_list:
            prompt = (
                f"Translate the English number word “{word}” into {lang}. "
                "Return ONLY the translation."
            )
            response = client.responses.create(
                model="gpt-4.1",
                input=prompt,
            )
            translated_response = response.output_text.strip()
            translations[word] = translated_response
            print(f"{word:<12} → {translated_response}")

        # build path in the tgt-language folder
        lang_cache = Path(OUTPUT_PATH, lang, "number_word_exact_map.json")
        lang_cache.parent.mkdir(parents=True, exist_ok=True)
        lang_cache.write_text(json.dumps(translations, ensure_ascii=False, indent=2))
        print(f"[{lang}] saved {len(translations)} items → {lang_cache}")

    # Get excel
    df = pd.DataFrame(
        {
            "english": list(translations.keys()),
            lang:      list(translations.values()),
        }
    )
    df.to_excel(excel_path, index=False)            # uses openpyxl backend
    print(f"[{lang}] Excel sheet written → {excel_path}")


