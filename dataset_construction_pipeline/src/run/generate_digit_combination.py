import yaml
import json
import argparse as arg
from typing import List, Dict, Tuple, Set
import numpy as np
from digit_generator.digit_generator import SmartDigitGenerator


def parse_yaml_config(yaml_path: str) -> Dict:
    """Parse the YAML configuration file"""
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
        
    # Load equations from JSON file
    with open(config['equation_file_path'], 'r') as file:
        equations = json.load(file)
        
    return {
        'equations': equations,
        'number_generation': config['number_generation'],
        'minimum_number': config['minimum_number'],
        'maximum_number': config['maximum_number']
    }


def process_all_equations(yaml_path: str, generator: SmartDigitGenerator) -> List[Dict]:
    """Process all equations from YAML config"""
    config = parse_yaml_config(yaml_path)

    success_count = 0
    total_count = 0
        
    output = []
        
    for idx, eq_data in enumerate(config['equations']):
        equation = eq_data['equation']
        restriction_list = eq_data.get("restriction", [])
        if restriction_list is None:
            restriction_list = []
        construction_list = eq_data.get("construction", [])
        if construction_list is None:
            construction_list = []
        
            
        combinations = generator.generate_single_question_digit_combination(
            equation=equation,
            minimum_number=config['minimum_number'],
            maximum_number=config['maximum_number'],
            total_combination_number=config['number_generation'],
            restriction_list=restriction_list,
            construction_list=construction_list
        )


        if(len(combinations) == config['number_generation']):
            success_count = success_count + 1
        
        print(f"current index{total_count}, number generated {len(combinations)}")


        total_count = total_count + 1

        

        output.append({
            "id": idx,
            **combinations
        })
    print(f"success {success_count} out of {total_count}")
    return output
    

def save_output(output: List[Dict], filepath: str):
    """Save output to JSON file"""
    with open(filepath, 'w') as file:
        json.dump(output, file, indent=2)




def main():

    parser = arg.ArgumentParser(description='generate the 10 digit combinations')
    parser.add_argument('--config', type=str, help='path to yaml file')
    args = parser.parse_args()  

    
    # Initialize generator
    generator = SmartDigitGenerator()


    output = process_all_equations(args.config, generator)
    #generator.save_output(output, '/home/mila/x/xut/github/distribution-dataset/result/digit_combination/digit_combination.json')
    generator.save_output(output, '/home/mila/x/xut/github/distribution-dataset/result/digit_combination/digit_combination_lower_maximum.json')

if __name__ == "__main__":
    main()