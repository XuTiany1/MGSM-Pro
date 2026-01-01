import yaml
import json
import random
import re
from fractions import Fraction
from typing import List, Dict, Tuple, Set
from itertools import combinations, product
import numpy as np

class SmartDigitGenerator:
    def __init__(self):
        self.used_combinations = set()
    
    def extract_digit_positions(self, equation: str) -> List[str]:
        """Extract digit placeholder positions from equation"""
        # Find patterns like {num_digit_1}, {num_digit_2}, etc.
        pattern = r'\{(num_digit_\d+)\}'
        positions = re.findall(pattern, equation)

        # Meaning only 1 digit is used
        if positions == []:
            pattern = r'\{(num_digit\d+)\}'
            positions = re.findall(pattern, equation)

        return positions
    
    def evaluate_equation(self, equation: str, digit_values: Dict[str, int]) -> float:
        """Evaluate equation with given digit values"""
        equation_copy = equation
        for key, value in digit_values.items():
            equation_copy = equation_copy.replace(f'{{{key}}}', str(value))
        
        try:
            # Evaluate the mathematical expression
            result = eval(equation_copy)
            return result
        except:
            return None
    
    def generate_single_question_digit_combination(
        self, 
        equation: str, 
        minimum_number: int, 
        maximum_number: int, 
        total_combination_number: int,
        restriction_list: List,
        construction_list: List
    ) -> Dict:
        """Generate digit combinations for a single equation"""
        
        # Extract digit positions from equation
        digit_positions = self.extract_digit_positions(equation)
        num_digits = len(digit_positions)
        
        if num_digits == 0:
            return {"error": "No digit placeholders found in equation"}
        
        combinations_list = []
        used_in_this_equation = set()

        # Strategy 3: Generate diverse random combinations
        remaining_count = total_combination_number - len(combinations_list)
        random_combinations = self._generate_random_combinations(
            equation, digit_positions,
            minimum_number, maximum_number,
            remaining_count,
            used_in_this_equation,
            restriction_list,
            construction_list
        )
        combinations_list.extend(random_combinations)
        
        # Format output
        result = {}
        for i, combo in enumerate(combinations_list, 1):
            combo_key = f"combination_{i}"
            result[combo_key] = {}
            
            # Add individual digits
            for j, (digit_name, digit_value) in enumerate(combo['digits'].items(), 1):
                result[combo_key][digit_name] = digit_value
            
            # Add answer
            result[combo_key]["answer"] = combo['answer']
        
        return result
    
    
    def _generate_random_combinations(
        self, equation, digit_positions,
        min_val, max_val, count, used_combinations, restriction_list, construction_list
    ) -> List[Dict]:
        """Generate random diverse combinations"""
        combinations = []
        
        # Create a distribution strategy
        for i in range(count):
            attempts = 0
            max_attempts = 1000000
            
            while attempts < max_attempts:
                attempts += 1

                # Build one value per unique placeholder (preserve first-seen order)
                unique_positions = list(dict.fromkeys(digit_positions))
                digits = [random.randint(min_val, max_val) for _ in range(len(unique_positions))]
                digit_values = dict(zip(unique_positions, digits))

                # Stable "used" key in the same order as unique_positions
                used_key = tuple(digit_values[p] for p in unique_positions)
                if used_key in used_combinations:
                    continue

                # Evaluate
                result = self.evaluate_equation(equation, digit_values)

                if result is not None and self._is_positive_int(result):

                    # Restriction double checking
                    ok = True
                    for r in (restriction_list or []):
                        if not self._restriction_ok(r, digit_values):
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    if construction_list != None:
                        digit_values = self._apply_constructions(construction_list, digit_values)
                        # Skip if construction is impossible
                    if digit_values == None:
                        continue
       
                    equation_copy = equation
                    for key, value in digit_values.items():
                        equation_copy = equation_copy.replace(f'{{{key}}}', str(value))
                    # Evaluate the mathematical expression
                    result = eval(equation_copy)


                    combinations.append({
                        "digits": digit_values.copy(),
                        "answer": int(result)
                    })
                    # remember this combo so we don't repeat it
                    used_combinations.add(used_key)
                    break

        return combinations
    
    def _apply_constructions(self, construction_list: list[str] | None, digit_values: dict[str, int]) -> dict[str, int]:
        """
        Each construction has the form:  {num_digit_4} = <arithmetic expression possibly using {num_digit_x}>
        Returns an updated copy of digit_values. If any construction is invalid, raises ValueError.
        """
        if not construction_list:
            return digit_values

        updated = dict(digit_values)  # work on a copy

        assign_pat = re.compile(r"^\s*\{(?P<lhs>[a-zA-Z0-9_]+)\}\s*=\s*(?P<rhs>.+?)\s*$")
        for stmt in construction_list:
            m = assign_pat.match(stmt)
            if not m:
                raise ValueError(f"Bad construction statement: {stmt!r}")

            lhs = m.group("lhs")
            rhs = m.group("rhs")

            # Compute RHS using current (possibly previously-constructed) values
            expr = self._render_with_values(rhs, updated)
            val = eval(expr)

            # Require positive integer (use caller's notion)
            if not self._is_positive_int(val):
                # If this is impossible construction
                print(f"Impossible construction: {val} is the constructed number")
                return None

            int_val = int(val)

            # If LHS already existed, ensure consistency
            if lhs in updated and updated[lhs] != int_val:
                raise ValueError(f"Inconsistent construction for {lhs}: {updated[lhs]} != {int_val}")

            updated[lhs] = int_val

        return updated

    def _render_with_values(self, text: str, vals: dict) -> str:
        out = text
        for k, v in vals.items():
            out = out.replace(f'{{{k}}}', str(v))
        return out

    def _restriction_ok(self, restriction: str, vals: dict) -> bool:
        _EQ_SINGLE_RE = re.compile(r'(?<![<>=!])=(?!=)')  # turn lone '=' into '=='

        # 1) substitute numbers
        expr = self._render_with_values(restriction, vals)
        # 2) normalize single '=' into '==', leaving <=, >=, !=, == intact
        expr = _EQ_SINGLE_RE.sub('==', expr)
        try:
            return bool(eval(expr))
        except Exception:
            return False


    def _is_positive_int(self, x):
        if x is None or isinstance(x, bool):
            return False
        if isinstance(x, int):
            return x > 0
        if isinstance(x, float):
            return x > 0 and x.is_integer()
        if isinstance(x, Fraction):
            return x > 0 and x.denominator == 1
        return False


    
    def process_all_equations(self, yaml_path: str) -> List[Dict]:
        """Process all equations from YAML config"""
        config = self.parse_yaml_config(yaml_path)
        
        output = []
        
        for idx, eq_data in enumerate(config['equations']):
            equation = eq_data['equation']
            original_answer = eq_data['original_answer']
            
            combinations = self.generate_single_question_digit_combination(
                equation=equation,
                original_answer=original_answer,
                minimum_number=config['minimum_number'],
                maximum_number=config['maximum_number'],
                total_combination_number=config['number_generation']
            )
            
            output.append({
                "id": idx,
                **combinations
            })
        
        return output
    
    def save_output(self, output: List[Dict], filepath: str):
        """Save output to JSON file"""
        with open(filepath, 'w') as file:
            json.dump(output, file, indent=2)
