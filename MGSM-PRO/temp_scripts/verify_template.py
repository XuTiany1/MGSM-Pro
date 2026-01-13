import json
import re
from pathlib import Path
from collections import defaultdict

def extract_variables(template_str):
    """
    Extract all variables from a template string.
    Returns a sorted list of variable names (without values).
    E.g., "{num_digit_1, 16}" -> "num_digit_1"
    """
    if not template_str:
        return []
    
    # Find all {variable, value} patterns and extract just the variable name
    pattern = r'\{([^,]+),[^}]+\}'
    variables = re.findall(pattern, template_str)
    return sorted(variables)

def get_template_variables(template_data):
    """
    Extract all variables from all template fields in a template.
    Returns a dict with field names as keys and sorted variable lists as values.
    """
    fields_to_check = ['symbolic_template', 'ic_sentence_template', 'ic_template']
    variables = {}
    
    for field in fields_to_check:
        if field in template_data and template_data[field]:
            variables[field] = extract_variables(template_data[field])
    
    return variables

def load_template(file_path):
    """Load a JSON template file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def validate_templates(template_root, reference_lang='english'):
    """
    Validate that all language templates have the same variables as the reference language (English).
    
    Args:
        template_root: Path to the template directory
        reference_lang: The reference language (default: 'english')
    
    Returns:
        Dictionary with validation results
    """
    template_path = Path(template_root).expanduser()
    
    # Load reference templates (English)
    ref_lang_path = template_path / reference_lang
    if not ref_lang_path.exists():
        print(f"ERROR: Reference language '{reference_lang}' not found at {ref_lang_path}")
        return None
    
    # Load all reference templates
    ref_templates = {}
    ref_files = sorted(ref_lang_path.glob("*.json"))
    
    print(f"Loading reference templates from {reference_lang}...")
    for ref_file in ref_files:
        template_id = ref_file.stem  # e.g., "0000"
        template_data = load_template(ref_file)
        if template_data:
            ref_templates[template_id] = get_template_variables(template_data)
    
    print(f"Loaded {len(ref_templates)} reference templates")
    
    # Get all language folders except reference
    language_folders = [f for f in template_path.iterdir() 
                       if f.is_dir() and f.name != reference_lang]
    
    # Results storage
    results = {
        'reference_lang': reference_lang,
        'languages_checked': [],
        'total_templates': len(ref_templates),
        'mismatches': defaultdict(lambda: defaultdict(list)),  # {template_id: {lang: [issues]}}
        'missing_templates': defaultdict(list),  # {lang: [template_ids]}
        'summary': defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})
    }
    
    # Check each language
    for lang_folder in sorted(language_folders):
        lang_name = lang_folder.name
        results['languages_checked'].append(lang_name)
        print(f"\nValidating {lang_name}...")
        
        # Check each template
        for template_id, ref_vars in ref_templates.items():
            results['summary'][lang_name]['total'] += 1
            
            template_file = lang_folder / f"{template_id}.json"
            
            # Check if template exists
            if not template_file.exists():
                results['missing_templates'][lang_name].append(template_id)
                results['summary'][lang_name]['incorrect'] += 1
                continue
            
            # Load and compare
            template_data = load_template(template_file)
            if not template_data:
                results['missing_templates'][lang_name].append(template_id)
                results['summary'][lang_name]['incorrect'] += 1
                continue
            
            lang_vars = get_template_variables(template_data)
            
            # Compare variables for each field
            has_mismatch = False
            for field in ref_vars.keys():
                ref_field_vars = ref_vars[field]
                lang_field_vars = lang_vars.get(field, [])
                
                # Separate name variables from non-name variables
                ref_name_vars = [v for v in ref_field_vars if v.startswith('name_')]
                ref_non_name_vars = [v for v in ref_field_vars if not v.startswith('name_')]
                
                lang_name_vars = [v for v in lang_field_vars if v.startswith('name_')]
                lang_non_name_vars = [v for v in lang_field_vars if not v.startswith('name_')]
                
                # Check if language template has NO name variables when reference has at least one
                if ref_name_vars and not lang_name_vars:
                    has_mismatch = True
                    issue = {
                        'field': field,
                        'expected': ref_field_vars,
                        'found': lang_field_vars,
                        'missing': ref_name_vars,
                        'extra': [],
                        'issue_type': 'missing_all_name_variables'
                    }
                    results['mismatches'][template_id][lang_name].append(issue)
                    continue
                
                # For non-name variables, they must match exactly
                if ref_non_name_vars != lang_non_name_vars:
                    has_mismatch = True
                    
                    # Find differences in non-name variables only
                    missing = set(ref_non_name_vars) - set(lang_non_name_vars)
                    extra = set(lang_non_name_vars) - set(ref_non_name_vars)
                    
                    issue = {
                        'field': field,
                        'expected': ref_field_vars,
                        'found': lang_field_vars,
                        'missing': sorted(missing),
                        'extra': sorted(extra),
                        'issue_type': 'non_name_variable_mismatch'
                    }
                    results['mismatches'][template_id][lang_name].append(issue)
            
            if has_mismatch:
                results['summary'][lang_name]['incorrect'] += 1
            else:
                results['summary'][lang_name]['correct'] += 1
    
    return results

def print_results(results):
    """Print validation results in a readable format."""
    if not results:
        return
    
    print("\n" + "="*80)
    print("TEMPLATE VALIDATION RESULTS")
    print("="*80)
    print(f"Reference Language: {results['reference_lang']}")
    print(f"Total Templates: {results['total_templates']}")
    print(f"Languages Checked: {len(results['languages_checked'])}")
    print("="*80)
    
    # Summary by language
    print("\nSUMMARY BY LANGUAGE:")
    print("-"*80)
    for lang in sorted(results['languages_checked']):
        summary = results['summary'][lang]
        total = summary['total']
        correct = summary['correct']
        incorrect = summary['incorrect']
        percentage = (correct / total * 100) if total > 0 else 0
        
        status = "✓" if incorrect == 0 else "✗"
        print(f"{status} {lang:15s}: {correct:3d}/{total:3d} correct ({percentage:5.1f}%) | {incorrect:3d} mismatches")
    
    # Missing templates
    if results['missing_templates']:
        print("\n" + "="*80)
        print("MISSING TEMPLATES:")
        print("-"*80)
        for lang, template_ids in sorted(results['missing_templates'].items()):
            if template_ids:
                print(f"\n{lang}:")
                print(f"  Missing: {', '.join(sorted(template_ids))}")
    
    # Detailed mismatches
    if results['mismatches']:
        print("\n" + "="*80)
        print("DETAILED MISMATCHES:")
        print("="*80)
        
        for template_id in sorted(results['mismatches'].keys()):
            print(f"\n{'Template ' + template_id:^80}")
            print("-"*80)
            
            for lang in sorted(results['mismatches'][template_id].keys()):
                print(f"\n  Language: {lang}")
                
                for issue in results['mismatches'][template_id][lang]:
                    print(f"    Field: {issue['field']}")
                    
                    if issue['issue_type'] == 'missing_all_name_variables':
                        print(f"    ⚠️  CRITICAL: Template has NO name variables!")
                        print(f"    Expected at least: {[v for v in issue['expected'] if v.startswith('name_')]}")
                    else:
                        print(f"    Expected: {issue['expected']}")
                        print(f"    Found:    {issue['found']}")
                        
                        if issue['missing']:
                            print(f"    Missing variables: {issue['missing']}")
                        if issue['extra']:
                            print(f"    Extra variables:   {issue['extra']}")
                    print()

def export_results(results, output_file):
    """Export results to a JSON file for further analysis."""
    if not results:
        return
    
    # Convert defaultdict to regular dict for JSON serialization
    export_data = {
        'reference_lang': results['reference_lang'],
        'languages_checked': results['languages_checked'],
        'total_templates': results['total_templates'],
        'summary': dict(results['summary']),
        'missing_templates': dict(results['missing_templates']),
        'mismatches': {k: dict(v) for k, v in results['mismatches'].items()}
    }
    
    output_path = Path(output_file).expanduser()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults exported to: {output_path}")

if __name__ == "__main__":
    # Define paths
    TEMPLATE_ROOT = "~/github/MGSM-PRO/dataste_construction_tools/template"
    OUTPUT_FILE = "~/github/MGSM-PRO/dataste_construction_tools/validation_results.json"
    
    print("Template Variable Validator")
    print("="*80)
    print(f"Template Root: {TEMPLATE_ROOT}")
    print(f"Reference Language: english")
    print("="*80)
    
    # Run validation
    results = validate_templates(TEMPLATE_ROOT, reference_lang='english')
    
    if results:
        # Print results
        print_results(results)
        
        # Export to file
        export_results(results, OUTPUT_FILE)
        
        # Final summary
        total_mismatches = sum(1 for template_id in results['mismatches'] 
                              for lang in results['mismatches'][template_id])
        
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"Total templates with mismatches: {len(results['mismatches'])}")
        print(f"Total language-template mismatches: {total_mismatches}")
        
        if total_mismatches == 0:
            print("\n✓ All templates match the English reference!")
        else:
            print(f"\n✗ Found {total_mismatches} mismatches across languages")
            print(f"See details above or in {OUTPUT_FILE}")