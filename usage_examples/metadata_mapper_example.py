#!/usr/bin/env python3
"""
Example usage of the metadata mapper tool
"""

import os
from geomapr.metadata_mapper import MetadataMapper

def main():
    # Check if credentials exist
    if not os.path.exists("creds.yaml"):
        print("Please create creds.yaml based on example_creds.yaml")
        print("You need an OpenRouter API key from https://openrouter.ai/keys")
        return
    
    # Initialize the mapper
    mapper = MetadataMapper()
    
    # Process the example files
    input_csv = "usage_examples/example_input_metadata.csv"
    target_csv = "usage_examples/example_target_template.csv"
    output_csv = "usage_examples/mapped_output.csv"
    
    if not os.path.exists(input_csv):
        print(f"Input file {input_csv} not found")
        return
    
    if not os.path.exists(target_csv):
        print(f"Target template {target_csv} not found")
        return
    
    print("Starting metadata mapping process...")
    print(f"Input: {input_csv}")
    print(f"Target: {target_csv}")
    print(f"Output: {output_csv}")
    
    try:
        mapper.process_csv_mapping(input_csv, target_csv, output_csv)
        print(f"\nSuccess! Check {output_csv} for results")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
