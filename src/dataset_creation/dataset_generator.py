import json
import random
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import re


class DatasetGenerator:
    """
    A class to generate multilingual math problem datasets with configurable name and number substitutions.
    """
    
    SUPPORTED_LANGUAGES = [
        "amharic", "chinese", "english", "french", "igbo", "japanese", "swahili", "twi", "yoruba"
    ]
    
    def __init__(self, tools_root: str):
        """
        Initialize the dataset generator.
        
        Args:
            tools_root: Root directory containing the dataset construction tools
        """
        self.tools_root = Path(tools_root)
        self.template_dir = self.tools_root / "template"
        self.names_dir = self.tools_root / "native_names"
        self.combinations_dir = self.tools_root / "numerical_combination" / "combination"
        
        # Cache for loaded data
        self.templates_cache = {}
        self.names_cache = {}
        self.combinations_cache = {}
    
    def load_template(self, template_id: str, language: str) -> Dict[str, Any]:
        """Load a template for a specific language."""
        cache_key = f"{language}_{template_id}"
        if cache_key not in self.templates_cache:
            template_path = self.template_dir / language / f"{template_id}.json"
            with open(template_path, 'r', encoding='utf-8') as f:
                self.templates_cache[cache_key] = json.load(f)
        return self.templates_cache[cache_key]
    
    def load_names(self, language: str) -> Dict[str, List[str]]:
        """Load names for a specific language."""
        if language not in self.names_cache:
            names_path = self.names_dir / f"{language}_words.json"
            with open(names_path, 'r', encoding='utf-8') as f:
                self.names_cache[language] = json.load(f)
        return self.names_cache[language]
    
    def load_combinations(self, template_id: str) -> Optional[List[Dict[str, Any]]]:
        """Load numerical combinations for a template. Returns None if file doesn't exist."""
        if template_id not in self.combinations_cache:
            combo_path = self.combinations_dir / f"{template_id}.jsonl"
            if not combo_path.exists():
                return None
            combinations = []
            try:
                with open(combo_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        combinations.append(json.loads(line))
                self.combinations_cache[template_id] = combinations
            except Exception as e:
                print(f"Warning: Error loading combinations for {template_id}: {e}")
                return None
        return self.combinations_cache[template_id]
    
    def get_available_templates(self, language: str = "english") -> List[str]:
        """Get list of available template IDs."""
        template_lang_dir = self.template_dir / language
        return sorted([f.stem for f in template_lang_dir.glob("*.json")])
    
    def extract_original_values(self, text: str) -> str:
        """
        Extract original values from placeholders, removing the variable syntax.
        Converts {name_female, Janet} -> Janet and {num_digit_1, 16} -> 16
        
        Args:
            text: Text with placeholders like {name_female, Janet} or {num_digit_1, 16}
        
        Returns:
            Text with only original values
        """
        # Pattern matches both name and number placeholders
        pattern = r'\{[^,]+,\s*([^}]+)\}'
        return re.sub(pattern, r'\1', text)
    
    def substitute_names(self, text: str, names_dict: Dict[str, List[str]], 
                        name_mapping: Optional[Dict[str, str]] = None) -> Tuple[str, Dict[str, str]]:
        """
        Substitute name placeholders in text with actual names.
        
        Args:
            text: Text with name placeholders like {name_female, Janet}
            names_dict: Dictionary of name categories to name lists
            name_mapping: Existing mapping to ensure consistency
        
        Returns:
            Tuple of (substituted text, name mapping used)
        """
        if name_mapping is None:
            name_mapping = {}
        
        # Find all name placeholders
        pattern = r'\{(name_\w+(?:_\d+)?),\s*([^}]+)\}'
        
        def replace_name(match):
            placeholder = match.group(1)  # e.g., "name_female" or "name_female_1"
            original_name = match.group(2)  # e.g., "Janet"
            
            # Extract base category (e.g., "name_female" from "name_female_1")
            base_match = re.match(r'(name_\w+?)(?:_\d+)?$', placeholder)
            if base_match:
                base_category = base_match.group(1)
            else:
                base_category = placeholder
            
            if placeholder not in name_mapping:
                if base_category in names_dict:
                    name_mapping[placeholder] = random.choice(names_dict[base_category])
                else:
                    name_mapping[placeholder] = original_name
            
            return name_mapping[placeholder]
        
        substituted_text = re.sub(pattern, replace_name, text)
        return substituted_text, name_mapping
    
    def substitute_numbers(self, text: str, num_values: Dict[str, int]) -> str:
        """
        Substitute number placeholders in text with actual numbers.
        
        Args:
            text: Text with number placeholders like {num_digit_1, 16}
            num_values: Dictionary mapping variable names to values
        
        Returns:
            Text with numbers substituted
        """
        pattern = r'\{(num_\w+),\s*([^}]+)\}'
        
        def replace_num(match):
            var_name = match.group(1)
            original_value = match.group(2)
            
            if var_name in num_values:
                return str(num_values[var_name])
            else:
                return original_value
        
        return re.sub(pattern, replace_num, text)
    
    def generate_problem(self, template: Dict[str, Any], language: str,
                        replace_names: bool, replace_numbers: bool,
                        num_values: Optional[Dict[str, int]] = None,
                        name_mapping: Optional[Dict[str, str]] = None,
                        use_ic: bool = False) -> Tuple[str, int, Dict[str, str]]:
        """
        Generate a single problem instance.
        
        Returns:
            Tuple of (question text, answer, name mapping used)
        """
        # Choose template type
        template_key = "ic_template" if use_ic else "symbolic_template"
        question = template[template_key]
        answer = template["original_answer"]
        
        # Strategy: Process in order - first handle what we're replacing, then clean up
        if replace_names and replace_numbers:
            # Mode N#: Replace both
            if num_values:
                question = self.substitute_numbers(question, num_values)
                if "ans" in num_values:
                    answer = num_values["ans"]
            names_dict = self.load_names(language)
            question, name_mapping = self.substitute_names(question, names_dict, name_mapping)
        elif replace_names:
            # Mode N: Replace names, extract original numbers
            # First extract numbers to their original values
            question = re.sub(r'\{(num_\w+),\s*([^}]+)\}', r'\2', question)
            # Then replace names
            names_dict = self.load_names(language)
            question, name_mapping = self.substitute_names(question, names_dict, name_mapping)
        elif replace_numbers:
            # Mode #: Replace numbers, extract original names
            # First extract names to their original values
            question = re.sub(r'\{(name_\w+(?:_\d+)?),\s*([^}]+)\}', r'\2', question)
            # Then replace numbers
            if num_values:
                question = self.substitute_numbers(question, num_values)
                if "ans" in num_values:
                    answer = num_values["ans"]
        else:
            # No replacement: extract all original values
            question = self.extract_original_values(question)
        
        return question, answer, name_mapping if replace_names else {}
    
    def generate_dataset(self,
                        template_range: Optional[List[int]] = None,
                        template_type: str = "symbolic",
                        substitution_mode: str = "N#",
                        num_combinations: int = 1,
                        combination_instances: Optional[List[int]] = None,
                        languages: Optional[List[str]] = None,
                        output_root: str = "./output",
                        skip_indices: Optional[List[int]] = None,
                        consistent_names: bool = False) -> None:
        """
        Generate complete dataset based on specifications.
        
        Args:
            template_range: List of template IDs to use (e.g., [0, 1, 2, 3])
            template_type: "symbolic" or "ic"
            substitution_mode: "N" (names only), "#" (numbers only), or "N#" (both)
            num_combinations: Number of dataset variations to create
            combination_instances: Specific combination instances to use
            languages: List of languages to generate for
            output_root: Root directory for output
            skip_indices: List of template indices to skip (e.g., [184, 86, 153])
            consistent_names: If True, use the same name substitutions across all instances
        """
        # Set defaults
        if languages is None:
            languages = self.SUPPORTED_LANGUAGES
        
        if template_range is None:
            template_range = [int(t) for t in self.get_available_templates("english")]
        
        # Convert skip_indices to set for faster lookup
        skip_set = set(skip_indices) if skip_indices else set()
        
        # Remove skipped indices from template_range
        if skip_set:
            template_range = [t for t in template_range if t not in skip_set]
            print(f"Skipping templates: {sorted(skip_set)}")
        
        # Parse substitution mode
        replace_names = "N" in substitution_mode
        replace_numbers = "#" in substitution_mode
        use_ic = template_type.lower() == "ic"
        
        # Set combination instances
        if combination_instances is None:
            combination_instances = list(range(1, num_combinations + 1))
        
        output_root = Path(output_root)
        
        # Track errors
        errors = []
        skipped_templates = set()
        
        for lang in languages:
            print(f"Generating dataset for language: {lang}")
            
            # Create output directory
            lang_dir = output_root / lang / template_type.upper()
            lang_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize name mapping per template if consistent_names is True
            # Key: template_id, Value: name_mapping dict
            template_name_mappings = {} if consistent_names else None
            
            # Generate each distribution dataset
            for dist_idx, combo_instance in enumerate(combination_instances):
                output_file = lang_dir / f"distribution_dataset_{dist_idx}.jsonl"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    for template_id in template_range:
                        template_id_str = f"{template_id:04d}"
                        
                        try:
                            # Load template
                            template = self.load_template(template_id_str, lang)
                            
                            # Get numerical values if needed
                            num_values = None
                            if replace_numbers:
                                combinations = self.load_combinations(template_id_str)
                                if combinations is None:
                                    if template_id_str not in skipped_templates:
                                        error_msg = f"Template {template_id_str}: Missing combination file"
                                        errors.append(error_msg)
                                        skipped_templates.add(template_id_str)
                                        print(f"  ⚠ Skipping {template_id_str}: No combination file found")
                                    continue
                                
                                # Find the specific combination instance
                                combo_data = next((c for c in combinations 
                                                 if c["combination_instance"] == combo_instance), None)
                                if combo_data:
                                    num_values = combo_data["variable_values"]
                                else:
                                    if template_id_str not in skipped_templates:
                                        error_msg = f"Template {template_id_str}: Missing combination instance {combo_instance}"
                                        errors.append(error_msg)
                                        skipped_templates.add(template_id_str)
                                        print(f"  ⚠ Skipping {template_id_str}: Combination instance {combo_instance} not found")
                                    continue
                            
                            # Get name mapping for this template
                            if consistent_names:
                                # For the first instance, create new mappings
                                # For subsequent instances, reuse the mappings
                                if dist_idx == 0:
                                    # First instance: create new mapping
                                    current_name_mapping = None
                                else:
                                    # Subsequent instances: reuse mapping from first instance
                                    current_name_mapping = template_name_mappings.get(template_id)
                            else:
                                # Random names for each instance
                                current_name_mapping = None
                            
                            # Generate problem
                            question, answer, used_name_mapping = self.generate_problem(
                                template, lang, replace_names, replace_numbers,
                                num_values, current_name_mapping, use_ic
                            )
                            
                            # Store name mapping for first instance if consistent_names
                            if consistent_names and dist_idx == 0 and replace_names:
                                template_name_mappings[template_id] = used_name_mapping
                            
                            # Write to file
                            record = {
                                "id": template_id,
                                "instance": dist_idx,
                                "question": question,
                                "answer": answer
                            }
                            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        
                        except FileNotFoundError as e:
                            if template_id_str not in skipped_templates:
                                error_msg = f"Template {template_id_str}: File not found - {e}"
                                errors.append(error_msg)
                                skipped_templates.add(template_id_str)
                                print(f"  ⚠ Skipping {template_id_str}: {e}")
                        except Exception as e:
                            if template_id_str not in skipped_templates:
                                error_msg = f"Template {template_id_str}: {type(e).__name__} - {e}"
                                errors.append(error_msg)
                                skipped_templates.add(template_id_str)
                                print(f"  ⚠ Skipping {template_id_str}: {e}")
                
                print(f"  Created: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("Dataset generation complete!")
        if errors:
            print(f"\n⚠ Encountered {len(errors)} error(s) during generation:")
            print(f"Skipped templates: {sorted(skipped_templates)}")
            print("\nError details:")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✓ All templates processed successfully!")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    generator = DatasetGenerator(tools_root="../../data_construction_tool")
    
    # Example: Generate 5 IC datasets with both name and number substitution
    # for English and French, using templates 0-3
    generator.generate_dataset(
        template_range=[0, 1, 2, 3],
        template_type="ic",
        substitution_mode="N#",
        num_combinations=5,
        combination_instances=[1, 2, 3, 4, 5],
        languages=["english", "french"],
        output_root="./dataset/my_dataset",
        consistent_names=True  # Same names across all instances
    )