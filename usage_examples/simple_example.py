#!/usr/bin/env python3

"""
Simple example of GeoMapr for GEO/SRA metadata retrieval
No credentials or external services required!
"""

from geomapr import MetadataProcessor

def main():
    print("🧬 GeoMapr - Simple GEO/SRA Metadata Retrieval")
    print("=" * 50)
    
    # Initialize the processor
    processor = MetadataProcessor()
    
    # Test with GSE212964 (smaller dataset)
    geo_series = "GSE212964"
    output_file = f"{geo_series}_metadata.csv"
    
    print(f"📊 Processing {geo_series}...")
    print("⏳ This may take a moment...")
    
    try:
        # Process the series
        result_df = processor.process_geo_series(geo_series, output_file)
        
        print(f"\n✅ Success! Generated {len(result_df)} file records")
        print(f"📄 Saved to: {output_file}")
        
        # Show a preview
        if len(result_df) > 0:
            print(f"\n🔍 Preview (first 3 files):")
            preview_cols = ['sample_title', 'file_name', 'file_format']
            print(result_df[preview_cols].head(3).to_string(index=False))
            
            print(f"\n📈 Summary:")
            print(f"  • Total files: {len(result_df)}")
            
            # Count file formats
            formats = result_df['file_format'].value_counts()
            print(f"  • File types: {len(formats)} different formats")
            
            # Show first few samples
            samples = result_df['sample_title'].nunique()
            print(f"  • Samples: {samples}")
            
        print(f"\n🎉 Complete! Check {output_file} for full results.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Try a different GEO series or check your internet connection")

if __name__ == "__main__":
    main()
