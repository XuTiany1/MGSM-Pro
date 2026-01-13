
"""
Now, I have the existing numerical combination template checked + verified. 
I now need to verify some previous digit combiantions that i have.. This is generated prior to my current method, I need to verify that the answer and num_digit_ like actually match the equation steps + restrictions

I will give as input two things
1. The original json file that contains all previous digit combination
/home/mila/x/xut/github/distribution-dataset/result/digit_combination/digit_combination.json
This contain things as such 
[
  {
    "id": 0,
    "combination_1": {
      "num_digit_1": 413,
      "num_digit_2": 187,
      "answer": 75922
    },
    "combination_2": {
      "num_digit_1": 498,
      "num_digit_2": 806,
      "answer": 395746
    },
    "combination_3": {
      "num_digit_1": 780,
      "num_digit_2": 521,
      "answer": 402733
    },
    "combination_4": {
      "num_digit_1": 817,
      "num_digit_2": 273,
      "answer": 221130
    },
    "combination_5": {
      "num_digit_1": 327,
      "num_digit_2": 423,
      "answer": 135360
    },
    "combination_6": {
      "num_digit_1": 759,
      "num_digit_2": 327,
      "answer": 245904
    },
    "combination_7": {
      "num_digit_1": 879,
      "num_digit_2": 351,
      "answer": 306072
    },
    "combination_8": {
      "num_digit_1": 44,
      "num_digit_2": 467,
      "answer": 17279
    },
    "combination_9": {
      "num_digit_1": 840,
      "num_digit_2": 689,
      "answer": 573937
    },
    "combination_10": {
      "num_digit_1": 215,
      "num_digit_2": 613,
      "answer": 127504
    }
  },
  {
    "id": 1,
    "combination_1": {
      "num_digit_1": 228,
      "answer": 342
    },
    "combination_2": {


2. The root folder path that contains the numerical templates 
/home/mila/x/xut/github/MGSM-PRO/dataste_construction_tools/numerical_combination/template
(reasoners) xut@cn-c028:~/github/MGSM-PRO/dataste_construction_tools/numerical_combination/template$ ls
0000.json  0014.json  0028.json  0042.json  0056.json  0070.json  0084.json  0099.json  0113.json  0127.json  0141.json  0155.json  0169.json  0183.json  0197.json  0211.json  0225.json  0239.json
0001.json  0015.json  etc....

Each json template looksl ike this
{
    "problem_id": "0002",
    "generation_order": ["num_digit_1", "num_digit_2", "num_digit_3", "num_ic"],
    "variables": {
        "num_digit_1": {"type": "int", "range": [50000, 100000]},
        "num_digit_2": {"type": "int", "range": [10000, 40000]},
        "num_digit_3": {"type": "int", "range": [100, 200]},
        "num_ic": {"type": "int", "range": [99, 1999]}
    },
    "equation_steps": {
        "num_digit_1": [],
        "num_digit_2": ["total_investment = num_digit_1 + num_digit_2"],
        "num_digit_3": [
            "increase_amount = num_digit_1 * (num_digit_3 / 100)",
            "final_value = num_digit_1 + increase_amount",
            "ans = final_value - total_investment"
        ]
    },
    "complexity_restrictions": {
        "ans": {"min_digits": 4, "max_digits": 6}
    },
    "restrictions": {
        "num_digit_1": [],
        "num_digit_2": [],
        "num_digit_3": [
            "increase_amount % 1 == 0",
            "ans > 0"
        ],
        "num_ic": []
    }
}

Here, we are interested in the equation steps + restrictions. 
Then ans in the template corrspond to the 'answer' in the digit_combination.json file

The id in the digit combinatoin.json correspond to the id.json file in the combination templates



What happens is the following:

It should be an automated python script. not calling any api. It shoul do the following
1. For each id, verify all the combinations there 
    - For each combination, take out num_digit_xxx and plug them into the equation steps in the numerical_template to get the ans
    - Verify that the ans is the same as the answer in the digit_combination.json
    - Check that the restrictions are followed. 
2. For each id, after verifying, flag any combination that is wrong. 
3. Provide an excel sheet at the end. 1 row per id. 1 column per combination. It should be True or False, with colors. Indicating if the id+combination is right according to the new numerical_combination template 
4. Also another excel sheet. This should tell me the id/combination and the reason it is wrong. If it is restriction violated (if so which one)? or if it is just answer equation 


"""





import json
import os
import math
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple

class CombinationVerifier:
    def __init__(self, combinations_file: str, template_folder: str, output_dir: str = "verification_results"):
        """
        Initialize the verifier.
        
        Args:
            combinations_file: Path to digit_combination.json
            template_folder: Path to folder containing 0000.json, etc.
            output_dir: Folder to save the resulting Excel files
        """
        self.combinations_path = Path(combinations_file)
        self.template_path = Path(template_folder)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup safe execution environment
        self.safe_builtins = {
            "abs": abs, "min": min, "max": max, "round": round,
            "sum": sum, "len": len, "int": int, "float": float,
            "sorted": sorted, "list": list, "range": range,
        }
        
    def load_json(self, path: Path) -> Any:
        with open(path, 'r') as f:
            return json.load(f)

    def get_template_path(self, problem_id: int) -> Path:
        """Convert integer ID (0) to zero-padded filename (0000.json)."""
        filename = f"{problem_id:04d}.json"
        return self.template_path / filename

    def evaluate_expression(self, expr: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate a mathematical expression."""
        safe_dict = {
            "__builtins__": self.safe_builtins,
            "math": math,
            "random": None # Disable random during verification to ensure determinism
        }
        safe_dict.update(context)
        return eval(expr, safe_dict)

    def execute_steps(self, steps: List[str], context: Dict[str, Any]):
        """Run equation steps to update context."""
        for step in steps:
            if "=" in step:
                var_name, expr = step.split("=", 1)
                var_name = var_name.strip()
                expr = expr.strip()
                context[var_name] = self.evaluate_expression(expr, context)

    def verify_single_instance(self, combination_data: Dict[str, Any], template: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify a single combination against the template logic.
        Returns: (is_valid, reason_message)
        """
        # 1. Initialize context with provided variables (num_digit_X)
        context = {}
        for k, v in combination_data.items():
            if k != 'answer': # Don't put the target answer in context yet
                context[k] = v
        
        target_answer = combination_data.get('answer')
        
        try:
            # 2. Re-run logic in generation order
            # Even though we have values, we must run steps to get intermediate variables 
            # (like 'total_investment') required for restrictions.
            generation_order = template.get("generation_order", [])
            equation_steps = template.get("equation_steps", {})
            restrictions = template.get("restrictions", {})
            
            for var_name in generation_order:
                # We skip generation (var is already in context), but we must run associated steps
                if var_name in equation_steps:
                    self.execute_steps(equation_steps[var_name], context)
                
                # Check restrictions associated with this variable
                if var_name in restrictions:
                    for rule in restrictions[var_name]:
                        if not self.evaluate_expression(rule, context):
                            return False, f"Restriction violated: {rule}"

            # 3. Check Answer
            if 'ans' not in context:
                return False, "Equation steps did not produce 'ans' variable"
            
            calculated_ans = context['ans']
            
            # Allow small float tolerance if necessary, otherwise exact match for ints
            is_match = False
            if isinstance(target_answer, int) and isinstance(calculated_ans, int):
                is_match = target_answer == calculated_ans
            else:
                is_match = math.isclose(target_answer, calculated_ans, rel_tol=1e-9)
            
            if not is_match:
                return False, f"Answer mismatch. Json: {target_answer}, Calculated: {calculated_ans}"
                
            return True, "Valid"

        except Exception as e:
            return False, f"Error during verification: {str(e)}"

    def run(self):
        print("Loading combinations...")
        combinations_list = self.load_json(self.combinations_path)
        
        matrix_data = [] # For the matrix excel
        details_data = [] # For the detailed logs
        
        for entry in combinations_list:
            p_id = entry.get('id')
            template_file = self.get_template_path(p_id)
            
            if not template_file.exists():
                print(f"Warning: Template for ID {p_id} not found at {template_file}")
                continue
                
            template = self.load_json(template_file)
            
            # Prepare row for matrix
            matrix_row = {'Problem ID': p_id}
            
            # Iterate through keys like 'combination_1', 'combination_2'
            # Sort keys to ensure column order 1, 2, 3...
            comb_keys = sorted([k for k in entry.keys() if k.startswith('combination_')], 
                             key=lambda x: int(x.split('_')[1]))
            
            print(f"Verifying Problem ID {p_id} ({len(comb_keys)} combinations)...")

            for key in comb_keys:
                comb_data = entry[key]
                is_valid, reason = self.verify_single_instance(comb_data, template)
                
                # Add to matrix
                matrix_row[key] = is_valid
                
                # Add to details if wrong (or all, but requirement said 'flag any that is wrong')
                # I will add all to details for completeness, filter later if needed.
                details_data.append({
                    "Problem ID": p_id,
                    "Combination": key,
                    "Valid": is_valid,
                    "Reason": reason,
                    "Input Data": str(comb_data)
                })
            
            matrix_data.append(matrix_row)

        self.save_reports(matrix_data, details_data)

    def save_reports(self, matrix_data: List[Dict], details_data: List[Dict]):
        print("\nGenerating Excel reports...")
        
        # --- 1. Matrix Report ---
        df_matrix = pd.DataFrame(matrix_data)
        
        # Function to color cells
        def color_boolean(val):
            if val is True:
                return 'background-color: #c6efce; color: #006100' # Green
            elif val is False:
                return 'background-color: #ffc7ce; color: #9c0006' # Red
            return ''

        # Apply style
        # We only apply style to combination columns (skip Problem ID)
        comb_cols = [c for c in df_matrix.columns if c != 'Problem ID']
        styled_matrix = df_matrix.style.map(color_boolean, subset=comb_cols)
        
        matrix_path = self.output_dir / "verification_matrix.xlsx"
        styled_matrix.to_excel(matrix_path, index=False, engine='openpyxl')
        print(f"Matrix saved to: {matrix_path}")

        # --- 2. Details Report ---
        df_details = pd.DataFrame(details_data)
        
        # Simple coloring for the 'Valid' column in details too
        styled_details = df_details.style.map(color_boolean, subset=['Valid'])
        
        details_path = self.output_dir / "verification_details.xlsx"
        styled_details.to_excel(details_path, index=False, engine='openpyxl')
        print(f"Details saved to: {details_path}")

if __name__ == "__main__":
    # CONFIGURATION
    COMBINATIONS_FILE = "/home/mila/x/xut/github/distribution-dataset/result/digit_combination/digit_combination.json"
    TEMPLATE_FOLDER = "/home/mila/x/xut/github/MGSM-PRO/dataste_construction_tools/numerical_combination/template"
    
    verifier = CombinationVerifier(COMBINATIONS_FILE, TEMPLATE_FOLDER)
    verifier.run()


