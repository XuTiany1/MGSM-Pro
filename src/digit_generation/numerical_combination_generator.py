import json
import random
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple


class NumericalCombinationGenerator:
    """Generator for creating unique numerical combinations based on problem templates."""
    
    def __init__(self, template_path: str, output_path: str, max_trials: int = 10000):
        """
        Initialize the generator.
        
        Args:
            template_path: Path to folder containing template JSON files
            output_path: Path to folder for storing generated combinations
            max_trials: Maximum number of attempts before giving up
        """
        self.template_path = Path(template_path)
        self.output_path = Path(output_path)
        self.max_trials = max_trials
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def load_template(self, problem_id: str) -> Dict[str, Any]:
        """Load a problem template from JSON file."""
        template_file = self.template_path / f"{problem_id}.json"
        try:
            with open(template_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def load_existing_combinations(self, problem_id: str) -> Set[Tuple]:
        """Load existing combinations to ensure uniqueness."""
        output_file = self.output_path / f"{problem_id}.jsonl"
        existing = set()
        
        if output_file.exists():
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Create a tuple of variable values for uniqueness check
                        var_values = data['variable_values']
                        # Exclude num_ic from uniqueness check
                        key = tuple(sorted((k, v) for k, v in var_values.items() if k != 'num_ic'))
                        existing.add(key)
        
        return existing
    
    def save_combination(self, problem_id: str, instance_num: int, 
                        variable_values: Dict[str, Any]):
        """Save a generated combination to the output file."""
        output_file = self.output_path / f"{problem_id}.jsonl"
        
        combination = {
            "combination_instance": instance_num,
            "variable_values": variable_values
        }
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(combination) + '\n')
    
    def check_digit_count(self, value: float) -> int:
        """Count the number of digits in a number."""
        return len(str(int(abs(value))))
    
    def evaluate_expression(self, expr: str, context: Dict[str, Any]) -> Any:
        """Safely evaluate an expression with given context."""
        import math
        
        # Add commonly needed built-in functions
        safe_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            "len": len,
            "int": int,
            "float": float,
            "sorted": sorted,
            "list": list,
            "range": range,
        }
        
        safe_dict = {
            "__builtins__": safe_builtins,
            "random": random,
            "math": math
        }
        safe_dict.update(context)
        return eval(expr, safe_dict)
        
    def check_restrictions(self, restrictions: List[str], context: Dict[str, Any]) -> bool:
        """Check if all restrictions are satisfied."""
        for restriction in restrictions:
            try:
                if not self.evaluate_expression(restriction, context):
                    return False
            except Exception:
                return False
        return True
    
    def check_complexity_restrictions(self, restrictions: Dict[str, Dict], 
                                     context: Dict[str, Any]) -> bool:
        """Check if complexity restrictions are satisfied."""
        for var_name, limits in restrictions.items():
            if var_name not in context:
                continue
            
            value = context[var_name]
            digit_count = self.check_digit_count(value)
            
            if "min_digits" in limits and digit_count < limits["min_digits"]:
                return False
            if "max_digits" in limits and digit_count > limits["max_digits"]:
                return False
        
        return True
    
    def generate_variable(self, var_name: str, var_config: Dict[str, Any], 
                        context: Dict[str, Any]) -> Any:
        """Generate a value for a variable based on its configuration."""
        var_type = var_config["type"]
        
        if "construction" in var_config:
            # Construct from existing values
            construction_expr = var_config["construction"]
            # Extract the right side of the assignment
            if "=" in construction_expr:
                construction_expr = construction_expr.split("=", 1)[1].strip()
            return self.evaluate_expression(construction_expr, context)
        elif "range" in var_config:
            # Generate from range
            range_vals = var_config["range"]
            if var_type == "int":
                return random.randint(range_vals[0], range_vals[1])
            elif var_type == "float":
                value = random.uniform(range_vals[0], range_vals[1])
                # Apply precision if specified
                if "precision" in var_config:
                    precision = var_config["precision"]
                    value = round(value, precision)
                return value
        
        raise ValueError(f"Unknown configuration for variable {var_name}")
    
    def execute_equation_steps(self, steps: List[str], context: Dict[str, Any]):
        """Execute equation steps and update context."""
        for step in steps:
            if "=" in step:
                var_name, expr = step.split("=", 1)
                var_name = var_name.strip()
                expr = expr.strip()
                context[var_name] = self.evaluate_expression(expr, context)
    
    def generate_single_combination(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single numerical combination based on template."""
        context = {}
        generation_order = template["generation_order"]
        variables_config = template["variables"]
        equation_steps = template.get("equation_steps", {})
        restrictions = template.get("restrictions", {})
        complexity_restrictions = template.get("complexity_restrictions", {})
        
        for var_name in generation_order:
            var_config = variables_config[var_name]
            
            # Generate the variable and add to context
            value = self.generate_variable(var_name, var_config, context)
            context[var_name] = value
            
            if var_name in equation_steps:
                self.execute_equation_steps(equation_steps[var_name], context)
            
            if var_name in restrictions:
                if not self.check_restrictions(restrictions[var_name], context):
                    return None
            
            if not self.check_complexity_restrictions(complexity_restrictions, context):
                return None
        
        result = {}
        for var_name in generation_order:
            result[var_name] = context[var_name]
        
        if 'ans' in context:
            result['ans'] = int(context['ans'])
        
        return result
    
    def generate_combinations(self, problem_id: str, num_combinations: int) -> int:
        """
        Generate unique numerical combinations for a problem.
        
        Returns:
            Number of combinations successfully generated.
            Returns -1 if the template file does not exist.
        """
        template = self.load_template(problem_id)
        
        if template is None:
            print(f"Warning: Template for problem {problem_id} not found. Skipping.")
            return -1
            
        existing = self.load_existing_combinations(problem_id)
        
        start_instance = len(existing) + 1
        generated_count = 0
        trials = 0
        
        while generated_count < num_combinations and trials < self.max_trials:
            trials += 1
            result = self.generate_single_combination(template)
            
            if result is None:
                continue
            
            key = tuple(sorted((k, v) for k, v in result.items() if k != 'num_ic'))
            if key in existing:
                continue
            
            instance_num = start_instance + generated_count
            self.save_combination(problem_id, instance_num, result)
            existing.add(key)
            generated_count += 1
            
            if generated_count % 10 == 0:
                print(f"Generated {generated_count}/{num_combinations} combinations for problem {problem_id}")
        
        if generated_count < num_combinations:
            print(f"Warning: Only generated {generated_count}/{num_combinations} combinations for problem {problem_id} after {trials} trials")
        
        return generated_count
    
    def generate_batch(self, problem_ids: List[str], num_combinations: int) -> Dict[str, int]:
        """Generate combinations for multiple problems."""
        results = {}
        
        for pid in problem_ids:
            print(f"\n{'='*60}")
            print(f"Processing problem {pid}")
            print(f"{'='*60}")
            count = self.generate_combinations(pid, num_combinations)
            results[pid] = count
            
            if count == -1:
                print(f"Skipped problem {pid} (Missing template)")
            else:
                print(f"Completed problem {pid}: {count}/{num_combinations} combinations generated")
        
        return results