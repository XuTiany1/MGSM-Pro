import json
import os
import re
from pathlib import Path

def remove_number_brackets(template_str):
    """
    Convert {num_text_1, ሦስት} to just ሦስት
    Convert {num_text, ሦስት} to just ሦስት
    Convert {num_frac_1, 1/2} to just 1/2
    Convert {num_frac, 1/2} to just 1/2
    But KEEP {num_digit_1, 16} as is
    And KEEP {num_digit, 16} as is
    And KEEP {name_female, ጃኔት} as is
    And KEEP {num_ic, 41} as is
    """
    if not template_str:
        return template_str
    
    # Replace ONLY {num_text*, value} and {num_frac*, value} with just value
    # This matches:
    # - {num_text, value}
    # - {num_text_1, value}
    # - {num_text_2, value}
    # - {num_frac, value}
    # - {num_frac_1, value}
    # etc.
    # Keep everything else like {num_digit*, value}, {name_*, value}, {num_ic, value}
    pattern = r'\{(num_text[^,]*|num_frac[^,]*),\s*([^}]+)\}'
    return re.sub(pattern, r'\2', template_str)

def keep_symbolic_format(template_str):
    """Keep the template with {variable, value} format as is"""
    return template_str

def transform_template(old_template):
    """Transform old template format to new format"""
    new_template = {
        "original_question": old_template["original_question"],
        "original_answer": old_template["original_answer"],
        "symbolic_template": remove_number_brackets(old_template["question_template"]),
        "ic_sentence_template": remove_number_brackets(old_template.get("ic_sentence", "")),
        "ic_template": remove_number_brackets(old_template.get("ic_question_template", "")),
        "equation": "UPDATE HERE"
    }
    
    return new_template

def migrate_templates(source_root, target_root):
    """
    Migrate templates from source to target directory
    
    Args:
        source_root: Path to ~/github/distribution-dataset/dataset_construction_tools/template
        target_root: Path to ~/github/MGSM-PRO/dataste_construction_tools/template
    """
    source_path = Path(source_root).expanduser()
    target_path = Path(target_root).expanduser()
    
    # Create target directory if it doesn't exist
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Get all language folders
    language_folders = [f for f in source_path.iterdir() if f.is_dir()]
    
    stats = {
        "total_files": 0,
        "successful": 0,
        "failed": 0,
        "languages": []
    }
    
    for lang_folder in language_folders:
        lang_name = lang_folder.name
        print(f"\nProcessing language: {lang_name}")
        stats["languages"].append(lang_name)
        
        # Create target language folder
        target_lang_path = target_path / lang_name
        target_lang_path.mkdir(parents=True, exist_ok=True)
        
        # Get all JSON files
        json_files = sorted(lang_folder.glob("*.json"))
        
        for json_file in json_files:
            stats["total_files"] += 1
            
            try:
                # Read old template
                with open(json_file, 'r', encoding='utf-8') as f:
                    old_template = json.load(f)
                
                # Transform to new format
                new_template = transform_template(old_template)
                
                # Write to new location
                target_file = target_lang_path / json_file.name
                with open(target_file, 'w', encoding='utf-8') as f:
                    json.dump(new_template, f, ensure_ascii=False, indent=2)
                
                stats["successful"] += 1
                print(f"  ✓ {json_file.name}")
                
            except Exception as e:
                stats["failed"] += 1
                print(f"  ✗ {json_file.name}: {str(e)}")
    
    # Print summary
    print("\n" + "="*60)
    print("MIGRATION SUMMARY")
    print("="*60)
    print(f"Languages processed: {', '.join(stats['languages'])}")
    print(f"Total files: {stats['total_files']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print("="*60)
    
    return stats

if __name__ == "__main__":
    # Define paths
    SOURCE_ROOT = "~/github/distribution-dataset/dataset_construction_tools/template"
    TARGET_ROOT = "~/github/MGSM-PRO/dataste_construction_tools/template"
    
    print("Template Migration Tool")
    print("="*60)
    print(f"Source: {SOURCE_ROOT}")
    print(f"Target: {TARGET_ROOT}")
    print("="*60)
    
    # Run migration
    stats = migrate_templates(SOURCE_ROOT, TARGET_ROOT)
    
    # Show example of transformation
    print("\n" + "="*60)
    print("EXAMPLE TRANSFORMATION")
    print("="*60)
    
    source_path = Path(SOURCE_ROOT).expanduser()
    example_file = source_path / "english" / "0000.json"
    
    if example_file.exists():
        with open(example_file, 'r', encoding='utf-8') as f:
            old = json.load(f)
        new = transform_template(old)
        
        print("\nOLD FORMAT (sample fields):")
        print(f"  - question_template: {old.get('question_template', '')[:80]}...")
        print(f"  - Keys: {list(old.keys())}")
        
        print("\nNEW FORMAT (sample fields):")
        print(f"  - symbolic_template: {new.get('symbolic_template', '')[:80]}...")
        print(f"  - Keys: {list(new.keys())}")