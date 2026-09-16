#!/usr/bin/env python3
"""
Script to generate numerical combinations for math problems.

Usage examples:
    python generate_combinations.py --problem_id 0184 --num_combinations 10
    python generate_combinations.py --problem_range 0 15 --num_combinations 10
"""

import argparse
from pathlib import Path
from numerical_combination_generator import NumericalCombinationGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate numerical combinations for math problems"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--problem_id",
        type=str,
        help="Single problem ID (e.g., '0000')"
    )
    group.add_argument(
        "--problem_range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of problem IDs (e.g., 0 15 for problems 0000-0015)"
    )
    
    parser.add_argument(
        "--num_combinations",
        type=int,
        required=True,
        help="Number of unique combinations to generate per problem"
    )
    
    parser.add_argument(
        "--template_path",
        type=str,
        default="../../data_construction_tool/numerical_combination/template",
        help="Path to folder containing template JSON files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="../../data_construction_tool/numerical_combination/combination",
        help="Path to folder for storing generated combinations"
    )
    
    parser.add_argument(
        "--max_trials",
        type=int,
        default=100000,
        help="Maximum number of trials before giving up (default: 100000)"
    )
    
    return parser.parse_args()


def generate_problem_id_list(start: int, end: int) -> list:
    """Generate list of problem IDs from range."""
    return [f"{i:04d}" for i in range(start, end + 1)]


def main():
    """Main execution function."""
    args = parse_arguments()
    
    if args.problem_id:
        problem_ids = [args.problem_id]
    else:
        start, end = args.problem_range
        problem_ids = generate_problem_id_list(start, end)
    
    generator = NumericalCombinationGenerator(
        template_path=args.template_path,
        output_path=args.output_path,
        max_trials=args.max_trials
    )
    
    print("="*60)
    print("NUMERICAL COMBINATION GENERATOR")
    print("="*60)
    print(f"Template Path: {args.template_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Problems to Process: {len(problem_ids)}")
    print(f"Combinations per Problem: {args.num_combinations}")
    print(f"Max Trials: {args.max_trials}")
    print("="*60)
    print()
    
    results = generator.generate_batch(problem_ids, args.num_combinations)
    
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    
    total_generated = 0
    total_requested_count = 0
    
    successful_ids = []
    skipped_ids = []
    incomplete_ids = []
    
    for pid in problem_ids:
        count = results.get(pid, 0)
        
        if count == -1:
            # -1 indicates missing template
            skipped_ids.append(pid)
            print(f"- Problem {pid}: SKIPPED (Template missing)")
        elif count == args.num_combinations:
            successful_ids.append(pid)
            total_generated += count
            total_requested_count += args.num_combinations
            print(f"✓ Problem {pid}: {count}/{args.num_combinations}")
        else:
            incomplete_ids.append(f"{pid} ({count}/{args.num_combinations})")
            total_generated += count
            total_requested_count += args.num_combinations
            print(f"⚠ Problem {pid}: {count}/{args.num_combinations} (Incomplete)")
    
    print("-" * 60)
    print(f"Successful: {len(successful_ids)}")
    
    print(f"Skipped (Missing Template): {len(skipped_ids)}")
    if skipped_ids:
        print(f"  IDs: {', '.join(skipped_ids)}")
        
    print(f"Incomplete/Failed: {len(incomplete_ids)}")
    if incomplete_ids:
        print(f"  IDs: {', '.join(incomplete_ids)}")
        
    print("-" * 60)
    
    # Calculate success rate only based on problems we actually attempted (i.e. templates existed)
    if total_requested_count > 0:
        success_rate = (total_generated / total_requested_count * 100)
        print(f"Total Generated: {total_generated}/{total_requested_count}")
        print(f"Success Rate (of attempted): {success_rate:.1f}%")
    else:
        print("No problems were attempted.")
        
    print("="*60)


if __name__ == "__main__":
    main()