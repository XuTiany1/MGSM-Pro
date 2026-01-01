import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
import pandas as pd

from typing import List, Dict, Any, Union, Optional
from evaluate import load

# top of file
import sys
try:
    sys.set_int_max_str_digits(20000)  # optional; can omit if using parse_int=str
except AttributeError:
    pass

# Questions to exclude (0-indexed)
EXCLUDED_QUESTIONS = {86, 91, 153, 184, 241}


def extract_final_number(response: str) -> Union[int, float]:
    """
    Extract last number from string, including numbers with spaces, commas, or $.
    """
    if not response or not response.strip():
        return 0

    # Pattern to match numbers including those with $, commas, spaces, decimals, and scientific notation
    pattern = re.compile(r'[-+]?\$?\d[\d ,]*(?:\.\d+)?(?:[eE][-+]?\d+)?')
    matches = pattern.findall(response)
    
    if not matches:
        return 0
    
    # Clean the last match
    num_str = matches[-1].replace(' ', '').replace(',', '').replace('$', '')
    
    if num_str in {'.', '+.', '-.'}:
        return 0

    try:
        # Try to parse as integer first
        if re.fullmatch(r'[-+]?\d+', num_str):
            return int(num_str)
        
        # Parse as decimal for precision
        d = Decimal(num_str)
        if d.is_infinite() or d.is_nan():
            return 0
        
        # Return as int if it's a whole number, otherwise as float
        return int(d) if d == d.to_integral() else float(d)
    except (InvalidOperation, OverflowError):
        return 0


def process_jsonl_file(exact_match_metric, file_path: str, cot_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Process a JSONL file and extract predictions for specified COT index or all COTs.
    
    Args:
        exact_match_metric: The metric to compute
        file_path: Path to the JSONL file
        cot_index: If specified (1-indexed), only process that COT. If None, process all COTs.
    """
    cot_preds: Dict[str, List[Union[int, float]]] = {}
    cot_refs:  Dict[str, List[Union[int, float, str]]] = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line, parse_int=str)  # big-int safe

                    # Check if this question should be excluded based on its 'id' field
                    question_id = data.get('id')
                    if int(question_id) in EXCLUDED_QUESTIONS:
                        print(f"Skipping excluded question with id={question_id} (line {line_num})")
                        continue

                    # Ground truth (keep as-is; we stringify later)
                    ground_truth = data.get('answer')

                    outputs = data.get('outputs', [])
                    if not isinstance(outputs, list):
                        outputs = []

                    # If cot_index is specified, only process that specific COT
                    if cot_index is not None:
                        if 0 < cot_index <= len(outputs):
                            cot_name = f"COT_{cot_index}"
                            if cot_name not in cot_preds:
                                cot_preds[cot_name] = []
                                cot_refs[cot_name] = []
                            
                            predicted = extract_final_number(outputs[cot_index - 1])
                            cot_preds[cot_name].append(predicted)
                            cot_refs[cot_name].append(ground_truth)
                    else:
                        # Process all COTs
                        for cot_idx, output in enumerate(outputs):
                            cot_name = f"COT_{cot_idx + 1}"
                            if cot_name not in cot_preds:
                                cot_preds[cot_name] = []
                                cot_refs[cot_name] = []

                            predicted = extract_final_number(output)
                            cot_preds[cot_name].append(predicted)
                            cot_refs[cot_name].append(ground_truth)

                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

    # Compute exact match per COT, with sanity check
    final_cot_result = {}
    for cot_name, preds in cot_preds.items():
        refs = cot_refs.get(cot_name, [])
        if len(preds) != len(refs):
            print(f"[WARN] {file_path} :: {cot_name} predictions={len(preds)} refs={len(refs)} — skipping mismatched tails")
            # If ever mismatched (shouldn't now), truncate to min length:
            n = min(len(preds), len(refs))
            preds = preds[:n]
            refs = refs[:n]

        curr = exact_match_metric.compute(
            predictions=[str(p) for p in preds],
            references=[str(r) for r in refs]
        )
        
        # Add count information
        curr['total_evaluated'] = len(preds)
        curr['excluded_count'] = len(EXCLUDED_QUESTIONS)
        
        final_cot_result[cot_name] = curr

    return final_cot_result


def list_jsonl_files_recursive(folder_path: str) -> List[Path]:
    root = Path(folder_path)
    return sorted(p for p in root.rglob("*.jsonl") if p.is_file())


def results_to_excel(exact_match_metric, folder_path: str, output_file: str, 
                     cot_index: Optional[int] = None, as_percent: bool = False, 
                     max_cots: int = 6) -> pd.DataFrame:
    """
    Build a table of COT scores per file and write to Excel.
    
    Args:
        exact_match_metric: The metric to compute
        folder_path: Path to folder containing JSONL files
        output_file: Path to output Excel file
        cot_index: If specified (1-indexed), only report that COT. If None, report all COTs up to max_cots.
        as_percent: If True, report as percentage (0-100), otherwise as decimal (0-1)
        max_cots: Maximum number of COTs to report (only used if cot_index is None)
    
    Returns:
        DataFrame with results
    """
    folder = Path(folder_path)
    files = list_jsonl_files_recursive(folder_path)

    if not files:
        print(f"No JSONL files found in {folder_path}")
        return pd.DataFrame()

    print(f"Excluding questions: {sorted(EXCLUDED_QUESTIONS)}")
    if cot_index is not None:
        print(f"Processing only COT {cot_index}")
    else:
        print(f"Processing all COTs (up to {max_cots})")
    print("=" * 60)

    rows = []
    for p in files:
        res = process_jsonl_file(exact_match_metric, str(p), cot_index=cot_index)
        row = {"file": str(p)}
        
        if cot_index is not None:
            # Only report the specified COT
            key_json = f"COT_{cot_index}"
            key_col = f"COT{cot_index}"
            val = res.get(key_json, None)

            if isinstance(val, dict) and "exact_match" in val:
                v = float(val["exact_match"])
            elif isinstance(val, (int, float)):
                v = float(val)
            else:
                v = None

            if v is not None and as_percent:
                v = round(100.0 * v, 1)
            row[key_col] = v
        else:
            # Report all COTs up to max_cots
            for i in range(1, max_cots + 1):
                key_json = f"COT_{i}"
                key_col = f"COT{i}"
                val = res.get(key_json, None)

                if isinstance(val, dict) and "exact_match" in val:
                    v = float(val["exact_match"])
                elif isinstance(val, (int, float)):
                    v = float(val)
                else:
                    v = None

                if v is not None and as_percent:
                    v = round(100.0 * v, 1)
                row[key_col] = v
        
        rows.append(row)

    df = pd.DataFrame(rows).set_index("file")

    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out)
    print(f"Saved Excel to: {out}")
    # also print for quick copy/paste
    print(df.to_string())
    return df


def main(folder_path: str, output_file: str, cot_index: Optional[int] = None):
    """
    Main function to process all JSONL files in a folder and save results.
    
    Args:
        folder_path: Path to folder containing JSONL files
        output_file: Name of output Excel file
        cot_index: If specified (1-indexed), only process that COT. If None, process all COTs.
    """
    print(f"Processing folder: {folder_path}")
    print("=" * 60)

    exact_match_metric = load("exact_match")
    
    if cot_index is not None:
        # Process only the specified COT
        results_to_excel(exact_match_metric, folder_path, output_file, 
                        cot_index=cot_index, as_percent=False)
    else:
        # Process all COTs (default to max 6)
        results_to_excel(exact_match_metric, folder_path, output_file, 
                        cot_index=None, as_percent=False, max_cots=6)


# Example usage
if __name__ == "__main__":
    folder_path = "/home/mila/x/xut/github/evaluation-pipeline/result/original_dataset/AFRI_MGSM"
    output_file = "/home/mila/x/xut/github/evaluation-pipeline/result/original_dataset/AFRI_MGSM/result.xlsx"
    
    # Option 1: Process only COT 3
    # main(folder_path, output_file, cot_index=3)
    
    # Option 2: Process all COTs (default)
    main(folder_path, output_file)