#!/usr/bin/env python3
"""
Script to run dataset generation with various configurations.
"""

from dataset_generator import DatasetGenerator
import argparse


def parse_template_range(range_str):
    """Parse template range from string like '0,1,2,3' or '0-5'"""
    if not range_str:
        return None
    
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in range_str.split(',')]


def parse_skip_indices(skip_str):
    """Parse skip indices from string"""
    if not skip_str:
        return None
    return [int(x.strip()) for x in skip_str.split(',')]


def parse_combination_instances(instances_str):
    """Parse combination instances from string"""
    if not instances_str:
        return None
    return [int(x.strip()) for x in instances_str.split(',')]


def parse_int_list(list_str):
    """Parse integer list from comma-separated string"""
    if not list_str:
        return None
    return [int(x.strip()) for x in list_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description='Generate multilingual math problem datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 5 IC datasets with name+number substitution for all languages
  python run_generator.py --template-type ic --mode N# --num-combos 5
  
  # Generate symbolic datasets with only name substitution for English
  python run_generator.py --template-type symbolic --mode N --languages english
  
  # Generate ONLY specific problem IDs (0, 1, 5, 10)
  python run_generator.py --only-problems 0,1,5,10 --mode N# --num-combos 10
  
  # Generate specific templates (0-3) with specific combination instances
  python run_generator.py --templates 0,1,2,3 --mode N# --instances 1,2,3,4,5
  
  # Generate for multiple specific languages
  python run_generator.py --languages english,french,chinese --mode N#
  
  # Skip specific problematic templates
  python run_generator.py --skip 86,153 --template-type ic --mode N# --num-combos 10


  

  python run_generator.py --only-problems 74,75,76,84,86,89,91,104,106,108,119,121,123,128,135,138,145,153,161,184,189,199,226,241,244 --mode N# --num-combos 10 --consistent-names --template-type ic



        """
    )
    
    parser.add_argument(
        '--tools-root',
        default='../../data_construction_tool',
        help='Root directory containing dataset construction tools (default: ../../data_construction_tool)'
    )
    
    parser.add_argument(
        '--templates',
        type=str,
        help='Template range: comma-separated (0,1,2,3) or range (0-3). If not specified, uses all templates.'
    )
    
    parser.add_argument(
        '--template-type',
        choices=['symbolic', 'ic'],
        default='symbolic',
        help='Template type: symbolic or ic (default: symbolic)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['N', '#', 'N#'],
        default='N#',
        help='Substitution mode: N (names only), # (numbers only), N# (both) (default: N#)'
    )
    
    parser.add_argument(
        '--num-combos',
        type=int,
        default=10,
        help='Number of dataset combinations to create (default: 1)'
    )
    
    parser.add_argument(
        '--instances',
        type=str,
        help='Specific combination instances to use (comma-separated, e.g., 1,2,3,4,5)'
    )
    
    parser.add_argument(
        '--skip',
        type=str,
        help='Template indices to skip (comma-separated, e.g., 184,86,153)'
    )
    
    parser.add_argument(
        '--only-problems',
        type=str,
        help='Generate ONLY these problem IDs (comma-separated, e.g., 0,1,5,10). Overrides --templates if specified.'
    )
    
    parser.add_argument(
        '--consistent-names',
        action='store_true',
        help='Use the same name substitutions across all instances (default: random names per instance)'
    )
    
    parser.add_argument(
        '--languages',
        type=str,
        help='Languages to generate (comma-separated). If not specified, generates for all supported languages.'
    )
    
    parser.add_argument(
        '--output',
        default='./dataset/output',
        help='Root output directory (default: ./dataset/output)'
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    template_range = parse_template_range(args.templates)
    combination_instances = parse_combination_instances(args.instances)
    skip_indices = parse_skip_indices(args.skip)
    only_problem_ids = parse_int_list(args.only_problems)
    
    # Handle --only-problems: it overrides --templates
    if only_problem_ids:
        template_range = only_problem_ids
        print(f"Using --only-problems: {only_problem_ids}")
        if args.templates:
            print("  (--templates argument ignored because --only-problems was specified)")
    
    languages = None
    if args.languages:
        languages = [lang.strip() for lang in args.languages.split(',')]
    
    # Validate languages
    if languages:
        supported = DatasetGenerator.SUPPORTED_LANGUAGES
        invalid = [lang for lang in languages if lang not in supported]
        if invalid:
            print(f"Error: Unsupported languages: {invalid}")
            print(f"Supported languages: {supported}")
            return
    
    # Initialize generator
    print("Initializing dataset generator...")
    generator = DatasetGenerator(tools_root=args.tools_root)
    
    # Print configuration
    print("\n" + "="*60)
    print("Dataset Generation Configuration")
    print("="*60)
    if only_problem_ids:
        print(f"Only Problems: {only_problem_ids}")
    else:
        print(f"Templates: {template_range if template_range else 'All available'}")
    print(f"Skip Indices: {skip_indices if skip_indices else 'None'}")
    print(f"Template Type: {args.template_type}")
    print(f"Substitution Mode: {args.mode}")
    print(f"Consistent Names: {'Yes' if args.consistent_names else 'No (random per instance)'}")
    print(f"Number of Combinations: {args.num_combos}")
    print(f"Combination Instances: {combination_instances if combination_instances else f'1-{args.num_combos}'}")
    print(f"Languages: {languages if languages else 'All supported languages'}")
    print(f"Output Directory: {args.output}")
    print("="*60 + "\n")
    
    # Generate dataset
    try:
        generator.generate_dataset(
            template_range=template_range,
            template_type=args.template_type,
            substitution_mode=args.mode,
            num_combinations=args.num_combos,
            combination_instances=combination_instances,
            languages=languages,
            output_root=args.output,
            skip_indices=skip_indices,
            consistent_names=args.consistent_names
        )
        print("\n✓ Dataset generation completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()