#!/usr/bin/env python3

"""
Example usage of the geomapr package
"""

from geomapr import MetadataProcessor

def main():
    """Example of using geomapr to process GEO series"""
    
    # Initialize the processor
    processor = MetadataProcessor()
    
    # Example 1: Get a summary of a GEO series
    print("=== Example 1: Series Summary ===")
    summary = processor.get_series_summary('GSE75748')
    
    print(f"Series: {summary['series_id']}")
    print(f"Title: {summary['title']}")
    print(f"Has SRA Data: {summary['has_sra_data']}")
    print(f"Total SRA Runs: {summary['total_sra_runs']}")
    
    # Example 2: Process a series with SRA data
    if summary['has_sra_data']:
        print(f"\n=== Example 2: Processing {summary['series_id']} ===")
        
        # Process and save to CSV
        result_df = processor.process_geo_series(
            summary['series_id'], 
            output_file=f"{summary['series_id']}_metadata.csv"
        )
        
        print(f"Generated CSV with {len(result_df)} rows")
        print("\nColumn names:")
        for i, col in enumerate(result_df.columns):
            print(f"  {i+1:2d}. {col}")
        
        print(f"\nFirst few entries:")
        preview_cols = ['run_accession', 'sample_accession', 'organism_name', 
                       'library_strategy', 'size_mb', 'sra_download_url']
        available_cols = [col for col in preview_cols if col in result_df.columns]
        print(result_df[available_cols].head(3).to_string(index=False))
    
    # Example 3: Try a series without SRA data
    print(f"\n=== Example 3: Series without SRA data ===")
    summary_no_sra = processor.get_series_summary('GSE212963')
    print(f"Series: {summary_no_sra['series_id']}")
    print(f"Title: {summary_no_sra['title']}")
    print(f"Has SRA Data: {summary_no_sra['has_sra_data']}")

if __name__ == "__main__":
    main()
