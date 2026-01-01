#!/usr/bin/env python3
import os, json
import pandas as pd
from pathlib import Path

def read_json_from_dir(directory_path: str):
    objs = []
    json_files = sorted(f for f in os.listdir(directory_path) if f.endswith(".json"))
    for fname in json_files:
        if fname == "ic_template.json":
            continue
        fpath = os.path.join(directory_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                objs.append((fname, json.load(f)))  # keep filename
        except Exception as e:
            print(f"[WARN] Failed reading {fname}: {e}")
    return objs

# --- CONFIG ---
LANGUAGE = "german"          # change me
TEMPLATE_DATASET = "symbolic"
INPUT_FOLDER = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/{TEMPLATE_DATASET}"
OUTPUT_EXCEL = f"/home/mila/x/xut/github/afrimgsm-symbolic/template/{LANGUAGE}/original_questions.xlsx"

# --- LOAD ---
json_objects = read_json_from_dir(INPUT_FOLDER)

# --- EXTRACT robustly ---
rows = []
for fname, obj in json_objects:
    if isinstance(obj, dict):
        q = obj.get("original_question")
        if q is not None:
            rows.append({"file": fname, "original_question": q})
        else:
            print(f"[WARN] {fname}: missing 'original_question'")
    elif isinstance(obj, list):
        found = False
        for i, item in enumerate(obj):
            if isinstance(item, dict) and "original_question" in item:
                rows.append({"file": f"{fname}#{i}", "original_question": item["original_question"]})
                found = True
        if not found:
            print(f"[WARN] {fname}: list with no 'original_question' entries")
    else:
        print(f"[WARN] {fname}: unsupported top-level type {type(obj)}")

# --- SAVE ---
df = pd.DataFrame(rows, columns=["file", "original_question"])
Path(os.path.dirname(OUTPUT_EXCEL)).mkdir(parents=True, exist_ok=True)
df.to_excel(OUTPUT_EXCEL, index=False)
print(f"[OK] Wrote {len(df)} rows to {OUTPUT_EXCEL}")
