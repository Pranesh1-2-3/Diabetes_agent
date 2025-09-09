import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.parser.parser import MedicalTableParser  # Updated to use MedicalTableParser

def load_ground_truth(json_path: str) -> dict:
    """Load ground truth annotations from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def validate_parser(annotations_dir: str, parser: MedicalTableParser) -> bool:
    """Validate parser against ground truth annotations.
    
    Args:
        annotations_dir: Directory containing annotation JSON files
        parser: Initialized TableParser instance
        
    Returns:
        bool: True if validation passes (≤1 field mismatch per image)
    """
    all_valid = True
    
    # Get all JSON files in annotations directory
    json_files = list(Path(annotations_dir).glob('*.json'))
    
    for json_file in json_files:
        print(f"\nValidating {json_file.name}...")
        
        # Load ground truth
        ground_truth = load_ground_truth(str(json_file))
        
        # Get the annotations directory absolute path
        annotations_dir_abs = os.path.abspath(annotations_dir)
        
        # Resolve the image path relative to the annotations directory
        image_path = os.path.normpath(os.path.join(os.path.dirname(annotations_dir_abs), ground_truth['image_path']))
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            all_valid = False
            continue
            
        # Parse image
        parsed_result = parser.parse_image(image_path)
        
        # Compare results
        mismatches = 0
        for field, expected in ground_truth['fields'].items():
            parsed = parsed_result.get(field)
            if parsed != expected:
                print(f"Mismatch in {field}:")
                print(f"  Expected: {expected}")
                print(f"  Got:      {parsed}")
                mismatches += 1
        
        # Check if validation passes
        if mismatches > 1:
            print(f"❌ Validation failed: {mismatches} mismatches")
            all_valid = False
        else:
            print(f"✓ Validation passed: {mismatches} mismatch")
    
    return all_valid

def main():
    parser = argparse.ArgumentParser(description='Validate table parser')
    parser.add_argument('--annotations', 
                      default='data/annotations',
                      help='Path to annotations directory')
    parser.add_argument('--advanced', 
                      action='store_true',
                      help='Use advanced Donut model')
    
    args = parser.parse_args()
    
    # Initialize parser
    table_parser = MedicalTableParser(use_advanced_model=args.advanced)
    
    # Run validation
    success = validate_parser(args.annotations, table_parser)
    
    # Exit with appropriate code
    exit(0 if success else 1)

if __name__ == '__main__':
    main()
