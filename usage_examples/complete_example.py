#!/usr/bin/env python3
"""
Complete example demonstrating the full geomapr workflow:
1. Retrieve metadata from GEO/SRA
2. Process and transform it
3. Map it to controlled vocabularies using AI
"""

import os
import pandas as pd
from geomapr.metadata_processor import GEOMetadataProcessor
from geomapr.metadata_mapper import MetadataMapper

def main():
    """Demonstrate the complete workflow"""
    
    print("=== GeoMapr Complete Workflow Example ===\n")
    
    # Step 1: Retrieve and process metadata (using existing functionality)
    print("Step 1: Retrieving metadata from GEO...")
    processor = GEOMetadataProcessor()
    
    # Example: Get metadata for a small dataset
    try:
        # This would normally retrieve from GEO, but for demo we'll use our example
        print("Using example input data for demonstration...")
        
        # Read our example input data
        input_df = pd.read_csv("usage_examples/example_input_metadata.csv")
        print(f"Loaded input metadata with {len(input_df)} rows")
        print("Input columns:", list(input_df.columns))
        
    except Exception as e:
        print(f"Error loading input data: {e}")
        return
    
    # Step 2: Check if we have the mapper credentials
    if not os.path.exists("creds.yaml"):
        print("\nStep 2: Setting up AI metadata mapper...")
        print("❌ Missing credentials file 'creds.yaml'")
        print("To complete this example, you need to:")
        print("1. Copy example_creds.yaml to creds.yaml")
        print("2. Add your OpenRouter API key (get one at https://openrouter.ai/keys)")
        print("\nSkipping AI mapping step for now...")
        return
    
    print("\nStep 2: AI-powered metadata mapping...")
    
    # Initialize the mapper
    try:
        mapper = MetadataMapper()
        print("✓ Metadata mapper initialized")
        
        # Check if we have target template
        if not os.path.exists("usage_examples/example_target_template.csv"):
            print("❌ Missing target template file")
            return
            
        target_df = pd.read_csv("usage_examples/example_target_template.csv")
        print(f"✓ Loaded target template with {len(target_df)} rows")
        print("Target columns:", list(target_df.columns))
        
        # Show what vocabularies are available
        print("\nAvailable controlled vocabularies:")
        vocabs = mapper.dictionary.extract_controlled_vocabularies()
        
        # Show relevant vocabularies for our target columns
        for col in target_df.columns:
            vocab = mapper.dictionary.get_vocabulary_for_column(col)
            if vocab:
                print(f"  {col}: {len(vocab)} terms available")
        
        print("\nStarting AI mapping process...")
        print("(This will require human input for each column)")
        
        # Run the mapping
        output_file = "usage_examples/demo_mapped_output.csv"
        mapper.process_csv_mapping(
            "usage_examples/example_input_metadata.csv",
            "usage_examples/example_target_template.csv", 
            output_file
        )
        
        print(f"\n✓ Mapping complete! Results saved to: {output_file}")
        
        # Show the results
        if os.path.exists(output_file):
            result_df = pd.read_csv(output_file)
            print("\nMapped results:")
            print(result_df.to_string(index=False))
            
    except Exception as e:
        print(f"Error in mapping process: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Workflow Complete ===")
    print("\nNext steps:")
    print("1. Review the mapped output file")
    print("2. Make any manual corrections needed")
    print("3. Use the standardized metadata for your analysis")


if __name__ == "__main__":
    main()
