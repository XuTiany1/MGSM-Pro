#!/usr/bin/env python3
"""
Script to validate numerical combination templates against original templates using Gemini API.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import google.generativeai as genai
from google.api_core import retry
from google import genai as new_genai
from google.genai import types
import pandas as pd


# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBFkWlvdk1WMCHDnp-CnAVzL5Q-a_G261k")
client = new_genai.Client(api_key=GOOGLE_API_KEY)

ORIGINAL_TEMPLATE_ROOT = "/home/mila/x/xut/github/MGSM-PRO/dataste_construction_tools/template/english"
NUMERICAL_TEMPLATE_ROOT = "/home/mila/x/xut/github/MGSM-PRO/dataste_construction_tools/numerical_combination/template"
OUTPUT_DIR = "./numerical_template_validation_results"


def is_retryable(exc):
    """Check if exception is retryable."""
    return isinstance(exc, Exception)


@retry.Retry(predicate=is_retryable,
            initial=1.0,
            multiplier=2.0,
            maximum=30.0,
            deadline=300.0)
def call_gemini_api(prompt: str, model_name: str = "gemini-2.0-flash-exp", max_tokens: int = 8000) -> str:
    """Call Gemini API with retry logic."""
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.0  # Use deterministic output for validation
        )
    )
    return response.text


def load_json_file(filepath: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_num_variables(symbolic_template: str) -> List[str]:
    """Extract all num_digit_X and num_ic variables from symbolic template."""
    import re
    # Pattern to match {num_digit_X, value} or {num_ic, value}
    pattern = r'\{(num_(?:digit_\d+|ic)),\s*[^}]+\}'
    matches = re.findall(pattern, symbolic_template)
    return list(set(matches))  # Remove duplicates


def construct_validation_prompt(original_data: Dict, numerical_data: Dict, problem_id: str) -> str:
    """Construct the validation prompt for Gemini."""
    
    original_answer = original_data.get("original_answer", "")
    symbolic_template = original_data.get("symbolic_template", "")
    
    prompt = f"""You are validating a numerical combination template for math problem generation.

**Problem ID:** {problem_id}

**Original Template Information:**
- Original Answer: {original_answer}
- Symbolic Template: {symbolic_template}

**Numerical Combination Template:**
```json
{json.dumps(numerical_data, indent=2)}
```

Please validate the following aspects and respond in JSON format:

1. **Variable Correspondence**: Do all num_* variables in the symbolic template exist in the numerical template's "variables" section, and vice versa? Are there any missing or extra variables? Ignore num_ic, this is an irrelevant number.

2. **Variable Definitions**: Do the ranges and constructions make sense for the problem context? Are they realistic and appropriate?

3. **Equation Steps**: Do the equation steps logically lead to the correct answer? Trace through the calculation with the original values from the symbolic template.

4. **Complexity Restrictions**: The original answer is {original_answer} which has {len(str(abs(int(original_answer))))} digits. The 'ans' complexity restriction should allow for {len(str(abs(int(original_answer))))}, {len(str(abs(int(original_answer))))+1}, or {len(str(abs(int(original_answer))))+2} digits. Does it?

5. **General Restrictions**: 
   - Must include "ans % 1 == 0" (ensures integer answer)
   - Must include "ans > 0" (ensures positive answer)
   - Other restrictions should be problem-specific and logical

Please respond in the following JSON format:
{{
  "variable_correspondence": true/false,
  "variable_correspondence_details": "explanation if false",
  "variable_definitions_valid": true/false,
  "variable_definitions_details": "explanation if false",
  "equation_steps_valid": true/false,
  "equation_steps_details": "explanation with calculation trace",
  "complexity_valid": true/false,
  "complexity_details": "explanation if false",
  "restrictions_valid": true/false,
  "restrictions_details": "explanation if false",
  "suggested_fix": {{
    // Only include if any validation failed
    // Provide corrected numerical template here
  }}
}}

Be thorough and precise in your validation."""

    return prompt


def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    """Parse Gemini's JSON response."""
    try:
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_text = response_text[start:end].strip()
        else:
            json_text = response_text.strip()
        
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        print(f"Response text: {response_text[:500]}...")
        return {
            "variable_correspondence": False,
            "variable_correspondence_details": f"Failed to parse response: {e}",
            "variable_definitions_valid": False,
            "variable_definitions_details": "Parse error",
            "equation_steps_valid": False,
            "equation_steps_details": "Parse error",
            "complexity_valid": False,
            "complexity_details": "Parse error",
            "restrictions_valid": False,
            "restrictions_details": "Parse error"
        }


def validate_single_template(problem_id: str, original_root: Path, numerical_root: Path) -> Dict[str, Any]:
    """Validate a single template pair."""
    
    print(f"Validating {problem_id}...")
    
    original_file = original_root / f"{problem_id}.json"
    numerical_file = numerical_root / f"{problem_id}.json"
    
    # Check if both files exist
    if not original_file.exists():
        return {
            "problem_id": problem_id,
            "status": "missing_original",
            "error": f"Original template file not found: {original_file}"
        }
    
    if not numerical_file.exists():
        return {
            "problem_id": problem_id,
            "status": "missing_numerical",
            "error": f"Numerical template file not found: {numerical_file}"
        }
    
    try:
        # Load both templates
        original_data = load_json_file(original_file)
        numerical_data = load_json_file(numerical_file)
        
        # Construct validation prompt
        prompt = construct_validation_prompt(original_data, numerical_data, problem_id)
        
        # Call Gemini API
        response_text = call_gemini_api(prompt)
        
        # Parse response
        validation_result = parse_gemini_response(response_text)
        
        # Add metadata
        validation_result["problem_id"] = problem_id
        validation_result["status"] = "validated"
        
        # Determine overall validity
        all_valid = (
            validation_result.get("variable_correspondence", False) and
            validation_result.get("variable_definitions_valid", False) and
            validation_result.get("equation_steps_valid", False) and
            validation_result.get("complexity_valid", False) and
            validation_result.get("restrictions_valid", False)
        )
        validation_result["overall_valid"] = all_valid
        
        return validation_result
        
    except Exception as e:
        return {
            "problem_id": problem_id,
            "status": "error",
            "error": str(e)
        }


def get_all_problem_ids(original_root: Path, numerical_root: Path) -> List[str]:
    """Get all problem IDs from both directories."""
    original_ids = set()
    numerical_ids = set()
    
    if original_root.exists():
        for file in original_root.glob("*.json"):
            original_ids.add(file.stem)
    
    if numerical_root.exists():
        for file in numerical_root.glob("*.json"):
            numerical_ids.add(file.stem)
    
    # Return union of both sets, sorted
    all_ids = original_ids.union(numerical_ids)
    return sorted(list(all_ids))


def save_results(results: List[Dict[str, Any]], output_dir: Path):
    """Save validation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_file = output_dir / "detailed_results.json"
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {detailed_file}")
    
    # Save summary
    summary_file = output_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        total = len(results)
        valid = sum(1 for r in results if r.get("overall_valid", False))
        invalid = total - valid
        
        f.write(f"Total Templates: {total}\n")
        f.write(f"Valid: {valid}\n")
        f.write(f"Invalid: {invalid}\n\n")
        
        f.write("="*80 + "\n")
        f.write("INVALID TEMPLATES\n")
        f.write("="*80 + "\n\n")
        
        for result in results:
            if not result.get("overall_valid", False):
                problem_id = result.get("problem_id", "unknown")
                f.write(f"\nProblem ID: {problem_id}\n")
                f.write("-" * 40 + "\n")
                
                if result.get("status") == "error":
                    f.write(f"Error: {result.get('error', 'Unknown error')}\n")
                    continue
                
                if not result.get("variable_correspondence", True):
                    f.write(f"❌ Variable Correspondence: {result.get('variable_correspondence_details', '')}\n")
                
                if not result.get("variable_definitions_valid", True):
                    f.write(f"❌ Variable Definitions: {result.get('variable_definitions_details', '')}\n")
                
                if not result.get("equation_steps_valid", True):
                    f.write(f"❌ Equation Steps: {result.get('equation_steps_details', '')}\n")
                
                if not result.get("complexity_valid", True):
                    f.write(f"❌ Complexity: {result.get('complexity_details', '')}\n")
                
                if not result.get("restrictions_valid", True):
                    f.write(f"❌ Restrictions: {result.get('restrictions_details', '')}\n")
                
                f.write("\n")
    
    print(f"Summary saved to: {summary_file}")
    
    # Save suggested fixes
    fixes = [r for r in results if "suggested_fix" in r and r["suggested_fix"]]
    if fixes:
        fixes_file = output_dir / "suggested_fixes.json"
        with open(fixes_file, 'w') as f:
            json.dump(fixes, f, indent=2)
        print(f"Suggested fixes saved to: {fixes_file}")
    
    # Save to Excel
    save_to_excel(results, output_dir)


def save_to_excel(results: List[Dict[str, Any]], output_dir: Path):
    """Save validation results to Excel file."""
    excel_file = output_dir / "validation_results.xlsx"
    
    # Prepare data for Excel
    excel_data = []
    for result in results:
        problem_id = result.get("problem_id", "unknown")
        status = result.get("status", "unknown")
        
        if status == "error":
            excel_data.append({
                "Problem ID": problem_id,
                "Status": "Error",
                "Overall Valid": False,
                "Variable Correspondence": False,
                "Variable Definitions": False,
                "Equation Steps": False,
                "Complexity": False,
                "Restrictions": False,
                "Error Details": result.get("error", ""),
                "Variable Correspondence Details": "",
                "Variable Definitions Details": "",
                "Equation Steps Details": "",
                "Complexity Details": "",
                "Restrictions Details": ""
            })
        elif status in ["missing_original", "missing_numerical"]:
            excel_data.append({
                "Problem ID": problem_id,
                "Status": status.replace("_", " ").title(),
                "Overall Valid": False,
                "Variable Correspondence": False,
                "Variable Definitions": False,
                "Equation Steps": False,
                "Complexity": False,
                "Restrictions": False,
                "Error Details": result.get("error", ""),
                "Variable Correspondence Details": "",
                "Variable Definitions Details": "",
                "Equation Steps Details": "",
                "Complexity Details": "",
                "Restrictions Details": ""
            })
        else:
            excel_data.append({
                "Problem ID": problem_id,
                "Status": "Validated",
                "Overall Valid": result.get("overall_valid", False),
                "Variable Correspondence": result.get("variable_correspondence", False),
                "Variable Definitions": result.get("variable_definitions_valid", False),
                "Equation Steps": result.get("equation_steps_valid", False),
                "Complexity": result.get("complexity_valid", False),
                "Restrictions": result.get("restrictions_valid", False),
                "Error Details": "",
                "Variable Correspondence Details": result.get("variable_correspondence_details", ""),
                "Variable Definitions Details": result.get("variable_definitions_details", ""),
                "Equation Steps Details": result.get("equation_steps_details", ""),
                "Complexity Details": result.get("complexity_details", ""),
                "Restrictions Details": result.get("restrictions_details", "")
            })
    
    # Create DataFrame
    df = pd.DataFrame(excel_data)
    
    # Create Excel writer with xlsxwriter engine for formatting
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Validation Results', index=False)
        
        # Get workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Validation Results']
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        
        true_format = workbook.add_format({
            'bg_color': '#90EE90',
            'border': 1
        })
        
        false_format = workbook.add_format({
            'bg_color': '#FFB6C1',
            'border': 1
        })
        
        text_format = workbook.add_format({
            'border': 1,
            'text_wrap': True,
            'valign': 'top'
        })
        
        # Set column widths
        worksheet.set_column('A:A', 15)  # Problem ID
        worksheet.set_column('B:B', 15)  # Status
        worksheet.set_column('C:H', 20)  # Boolean columns
        worksheet.set_column('I:I', 30)  # Error Details
        worksheet.set_column('J:N', 40)  # Detail columns
        
        # Format header row
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Format data rows
        for row_num in range(len(df)):
            # Problem ID and Status
            worksheet.write(row_num + 1, 0, df.iloc[row_num, 0], text_format)
            worksheet.write(row_num + 1, 1, df.iloc[row_num, 1], text_format)
            
            # Boolean columns with conditional formatting
            for col_num in range(2, 8):
                value = df.iloc[row_num, col_num]
                cell_format = true_format if value else false_format
                worksheet.write(row_num + 1, col_num, value, cell_format)
            
            # Detail columns
            for col_num in range(8, len(df.columns)):
                worksheet.write(row_num + 1, col_num, df.iloc[row_num, col_num], text_format)
        
        # Freeze first row
        worksheet.freeze_panes(1, 0)
    
    print(f"Excel results saved to: {excel_file}")
    
    # Also create a summary sheet
    summary_excel_file = output_dir / "validation_summary.xlsx"
    
    total = len(results)
    valid = sum(1 for r in results if r.get("overall_valid", False))
    invalid = total - valid
    missing_original = sum(1 for r in results if r.get("status") == "missing_original")
    missing_numerical = sum(1 for r in results if r.get("status") == "missing_numerical")
    errors = sum(1 for r in results if r.get("status") == "error")
    
    summary_data = {
        "Metric": [
            "Total Templates",
            "Valid Templates",
            "Invalid Templates",
            "Missing Original",
            "Missing Numerical",
            "Errors"
        ],
        "Count": [
            total,
            valid,
            invalid,
            missing_original,
            missing_numerical,
            errors
        ],
        "Percentage": [
            "100%",
            f"{valid/total*100:.1f}%" if total > 0 else "0%",
            f"{invalid/total*100:.1f}%" if total > 0 else "0%",
            f"{missing_original/total*100:.1f}%" if total > 0 else "0%",
            f"{missing_numerical/total*100:.1f}%" if total > 0 else "0%",
            f"{errors/total*100:.1f}%" if total > 0 else "0%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    with pd.ExcelWriter(summary_excel_file, engine='xlsxwriter') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        workbook = writer.book
        worksheet = writer.sheets['Summary']
        
        # Format
        worksheet.set_column('A:A', 25)
        worksheet.set_column('B:B', 15)
        worksheet.set_column('C:C', 15)
        
        # Header format
        for col_num, value in enumerate(summary_df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Data format
        for row_num in range(len(summary_df)):
            for col_num in range(len(summary_df.columns)):
                worksheet.write(row_num + 1, col_num, summary_df.iloc[row_num, col_num], text_format)
    
    print(f"Excel summary saved to: {summary_excel_file}")


def print_summary(results: List[Dict[str, Any]]):
    """Print validation summary to console."""
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    total = len(results)
    valid = sum(1 for r in results if r.get("overall_valid", False))
    invalid = total - valid
    missing_original = sum(1 for r in results if r.get("status") == "missing_original")
    missing_numerical = sum(1 for r in results if r.get("status") == "missing_numerical")
    errors = sum(1 for r in results if r.get("status") == "error")
    
    print(f"\nTotal Templates Checked: {total}")
    print(f"✓ Valid: {valid}")
    print(f"✗ Invalid: {invalid}")
    if missing_original > 0:
        print(f"⚠ Missing Original Template: {missing_original}")
    if missing_numerical > 0:
        print(f"⚠ Missing Numerical Template: {missing_numerical}")
    if errors > 0:
        print(f"⚠ Errors: {errors}")
    
    print("\n" + "="*80)
    print("INVALID TEMPLATE IDs")
    print("="*80)
    
    invalid_ids = [r["problem_id"] for r in results if not r.get("overall_valid", False)]
    if invalid_ids:
        for pid in invalid_ids:
            print(f"  - {pid}")
    else:
        print("  None! All templates are valid.")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("="*80)
    print("NUMERICAL TEMPLATE VALIDATOR")
    print("="*80)
    print(f"\nOriginal Template Root: {ORIGINAL_TEMPLATE_ROOT}")
    print(f"Numerical Template Root: {NUMERICAL_TEMPLATE_ROOT}")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Get all problem IDs
    original_root = Path(ORIGINAL_TEMPLATE_ROOT)
    numerical_root = Path(NUMERICAL_TEMPLATE_ROOT)
    problem_ids = get_all_problem_ids(original_root, numerical_root)
    
    print(f"\nFound {len(problem_ids)} problem IDs to validate")
    
    # Validate templates in parallel
    print("\nStarting validation...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        validate_func = partial(
            validate_single_template,
            original_root=original_root,
            numerical_root=numerical_root
        )
        results = list(executor.map(validate_func, problem_ids))
    
    # Save results
    output_dir = Path(OUTPUT_DIR)
    save_results(results, output_dir)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()