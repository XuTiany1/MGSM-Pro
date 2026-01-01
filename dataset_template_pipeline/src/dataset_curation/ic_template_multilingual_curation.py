import re
import os
import json
import torch
import requests
import pandas as pd
from pathlib import Path
from openai import OpenAI
from collections import Counter
from datasets import load_dataset
from typing import List, Dict, Optional, Union

#######################
# Helper Functions
#######################
def read_json_from_dir(directory_path: str):
    objects = []
    json_files = sorted(
        (f for f in os.listdir(directory_path)
         if f.endswith('.json') and f[:-5].isdigit()),
        key=lambda x: int(x[:-5])
    )
    for file_name in json_files:
        file_path = os.path.join(directory_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                # stash filename for traceability
                obj["_src_filename"] = file_name
                objects.append(obj)
        except Exception as e:
            print(f"failed reading {file_name}: {e}")
    return objects

#######################
# Hyperparams
#######################
LANG_CODE = "ibo"
LANGUAGE = "igbo"
INPUT_FOLDER = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/english/IC"
OUTPUT_PATH = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/IC"

INSTRUCT_FILE_FILLER = "/home/mila/x/xut/github/afrimgsm-symbolic/prompt/template_no_op_construction/template_translation_{lang}.txt"
INSTRUCT_FILE = INSTRUCT_FILE_FILLER.format(lang=LANG_CODE)

# --- OpenAI client (use env var!) ---
# export OPENAI_API_KEY=sk-...
client = OpenAI(api_key="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")
with open(INSTRUCT_FILE, "r", encoding="utf-8") as f:
    template_construction_instructions = f.read()
print("loaded openai client & instructions")

#######################
# IC sentence Translation Logic
#######################
src_json_objects = read_json_from_dir(INPUT_FOLDER)

rows = []  # one dict per IC row

# initialize rows with source info
for obj in src_json_objects:
    num_ic = obj.get("variable_number_dictionary", {}).get("num_ic", None)
    # Make num_ic serializable/compact for Excel
    if isinstance(num_ic, (list, tuple)):
        num_ic_cell = ", ".join(map(str, num_ic))
    else:
        num_ic_cell = str(num_ic) if num_ic is not None else ""

    rows.append({
        "source_file": obj.get("_src_filename", ""),
        "num_ic": num_ic_cell,
        "ic_question_template_en": obj.get("ic_question_template", ""),
        "ic_sentence_en": obj.get("ic_sentence", ""),
        # reserved columns for target
        f"ic_sentence_{LANG_CODE}": "",
    })

print(f"loaded {len(rows)} source sentences")

# translate each row’s ic_sentence_en → target
for i, r in enumerate(rows):
    curr_eng_ic_sentence = r["ic_sentence_en"]
    request = f"Input: {curr_eng_ic_sentence}\nOutput:"

    response = client.responses.create(
        model="gpt-4.1",
        instructions=template_construction_instructions,
        input=request,
    )
    tgt_text = response.output_text.strip()
    rows[i][f"ic_sentence_{LANG_CODE}"] = tgt_text
    print(f"[{i+1}/{len(rows)}] {tgt_text}")

# ── Save JSON (your original step) ──
os.makedirs(OUTPUT_PATH, exist_ok=True)
template_filename = "ic_template.json"
with open(os.path.join(OUTPUT_PATH, template_filename), "w", encoding="utf-8") as f:
    json.dump([r[f"ic_sentence_{LANG_CODE}"] for r in rows], f, ensure_ascii=False, indent=2)

# ── Save Excel ──
excel_path = os.path.join(OUTPUT_PATH, f"ic_template_{LANG_CODE}.xlsx")
df = pd.DataFrame(rows, columns=[
    "source_file",
    "num_ic",
    "ic_question_template_en",
    "ic_sentence_en",
    f"ic_sentence_{LANG_CODE}",
])
df.to_excel(excel_path, index=False)
print(f"Wrote Excel to: {excel_path}")
